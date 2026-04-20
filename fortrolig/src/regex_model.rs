//! Regex-based PII detection engine.
//!
//! Provides fast, thread-safe detection of personally identifiable information
//! in text using pre-compiled regex patterns. All patterns are compiled once on
//! first use and shared safely across threads via [`LazyLock`].
//!
//! The public API is a single function [`detect_pii`] — no model object needed.

use std::collections::HashSet;
use std::net::IpAddr;
use std::sync::LazyLock;

use fancy_regex::Regex;

// ── PII Labels ──────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PiiLabel {
    BankAccount,
    DriversLicense,
    EmailAddress,
    IpAddress,
    Passport,
    Pci,
    PhoneNumber,
    PreciseLocation,
    Ssn,
    VehicleId,
}

impl PiiLabel {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::BankAccount => "BANK_ACCOUNT",
            Self::DriversLicense => "DRIVERS_LICENSE",
            Self::EmailAddress => "EMAIL_ADDRESS",
            Self::IpAddress => "IP_ADDRESS",
            Self::Passport => "PASSPORT",
            Self::Pci => "PCI",
            Self::PhoneNumber => "PHONE_NUMBER",
            Self::PreciseLocation => "PRECISE_LOCATION",
            Self::Ssn => "SSN",
            Self::VehicleId => "VEHICLE_ID",
        }
    }

    pub fn from_label_str(s: &str) -> Option<Self> {
        match s {
            "BANK_ACCOUNT" => Some(Self::BankAccount),
            "DRIVERS_LICENSE" => Some(Self::DriversLicense),
            "EMAIL_ADDRESS" => Some(Self::EmailAddress),
            "IP_ADDRESS" => Some(Self::IpAddress),
            "PASSPORT" => Some(Self::Passport),
            "PCI" => Some(Self::Pci),
            "PHONE_NUMBER" => Some(Self::PhoneNumber),
            "PRECISE_LOCATION" => Some(Self::PreciseLocation),
            "SSN" => Some(Self::Ssn),
            "VEHICLE_ID" => Some(Self::VehicleId),
            _ => None,
        }
    }
}

impl std::fmt::Display for PiiLabel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

pub const REGEX_COMPATIBLE_LABELS: &[PiiLabel] = &[
    PiiLabel::BankAccount,
    PiiLabel::DriversLicense,
    PiiLabel::EmailAddress,
    PiiLabel::IpAddress,
    PiiLabel::Passport,
    PiiLabel::Pci,
    PiiLabel::PhoneNumber,
    PiiLabel::PreciseLocation,
    PiiLabel::Ssn,
    PiiLabel::VehicleId,
];

// ── Span ────────────────────────────────────────────────────────────

/// A detected PII region within the input text (byte offsets, UTF-8).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Span {
    pub start: usize,
    pub end: usize,
    pub label: PiiLabel,
}

// ── String Utilities ────────────────────────────────────────────────

/// Strip whitespace and hyphens.
fn compact(value: &str) -> String {
    value
        .chars()
        .filter(|c| !c.is_whitespace() && *c != '-')
        .collect()
}

/// Extract ASCII digits only.
fn ascii_digits(value: &str) -> String {
    value.chars().filter(|c| c.is_ascii_digit()).collect()
}

// ── Validators ──────────────────────────────────────────────────────

fn is_luhn_valid(number: &str) -> bool {
    let ds: Vec<u32> = number.chars().filter_map(|c| c.to_digit(10)).collect();
    if ds.is_empty() {
        return false;
    }
    let parity = ds.len() % 2;
    let checksum: u32 = ds
        .iter()
        .enumerate()
        .map(|(i, &d)| {
            if i % 2 == parity {
                let d2 = d * 2;
                if d2 > 9 { d2 - 9 } else { d2 }
            } else {
                d
            }
        })
        .sum();
    checksum % 10 == 0
}

fn is_valid_ip(value: &str) -> bool {
    value.trim().parse::<IpAddr>().is_ok()
}

fn is_valid_iban(value: &str) -> bool {
    let c = compact(value).to_ascii_uppercase();
    c.len() >= 15
        && c.len() <= 34
        && c.as_bytes()
            .get(..2)
            .is_some_and(|b| b.iter().all(|x| x.is_ascii_alphabetic()))
        && c.as_bytes()
            .get(2..4)
            .is_some_and(|b| b.iter().all(|x| x.is_ascii_digit()))
        && c.chars().all(|ch| ch.is_ascii_alphanumeric())
}

fn is_valid_account_number(value: &str) -> bool {
    let c = compact(value).to_ascii_uppercase();
    let digit_count = ascii_digits(&c).len();
    is_valid_iban(value)
        || (c.chars().all(|ch| ch.is_ascii_alphanumeric()) && (4..=18).contains(&digit_count))
}

fn is_valid_phone(value: &str) -> bool {
    let d = ascii_digits(value);
    if !(8..=15).contains(&d.len()) {
        return false;
    }
    if (13..=19).contains(&d.len()) && is_luhn_valid(&d) {
        return false;
    }
    value.contains(['+', '-', '(', ')', ' ']) || d.len() == 8
}

fn is_leap_year(year: u32) -> bool {
    (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0)
}

fn is_valid_date(year: u32, month: u32, day: u32) -> bool {
    if !(1..=12).contains(&month) || day == 0 {
        return false;
    }
    let max_day = match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 => {
            if is_leap_year(year) {
                29
            } else {
                28
            }
        }
        _ => return false,
    };
    day <= max_day
}

fn is_valid_cpr(value: &str) -> bool {
    let c = ascii_digits(value);
    if c.len() != 10 {
        return false;
    }
    let day: u32 = c[..2].parse().unwrap_or(0);
    let month: u32 = c[2..4].parse().unwrap_or(0);
    let year_suffix: u32 = c[4..6].parse().unwrap_or(0);
    [1900u32, 2000]
        .iter()
        .any(|&century| is_valid_date(century + year_suffix, month, day))
}

fn is_valid_passport(value: &str) -> bool {
    let c = compact(value).to_ascii_uppercase();
    let digit_count = ascii_digits(&c).len();
    let letter_count = c.chars().filter(|ch| ch.is_ascii_alphabetic()).count();
    c.chars().all(|ch| ch.is_ascii_alphanumeric())
        && (6..=12).contains(&c.len())
        && (5..=10).contains(&digit_count)
        && letter_count <= 4
}

fn is_valid_passport_simple(value: &str) -> bool {
    let c = compact(value);
    let digit_count = ascii_digits(&c).len();
    c.chars().all(|ch| ch.is_ascii_alphanumeric()) && (6..=9).contains(&digit_count)
}

fn is_valid_drivers_license(value: &str) -> bool {
    let c = compact(value).to_ascii_uppercase();
    let digit_count = ascii_digits(&c).len();
    c.chars().all(|ch| ch.is_ascii_alphanumeric())
        && (5..=20).contains(&c.len())
        && digit_count >= 1
        && c.chars().any(|ch| ch.is_ascii_alphabetic())
}

fn is_valid_dl_10char(value: &str) -> bool {
    ascii_digits(value).len() >= 1 && value.chars().filter(|c| c.is_ascii_alphabetic()).count() >= 3
}

fn is_valid_pci_luhn(value: &str) -> bool {
    let d = ascii_digits(value);
    (13..=19).contains(&d.len())
        && (is_luhn_valid(&d)
            || d.starts_with('4')
            || d.starts_with('5')
            || d.starts_with('6')
            || d.starts_with("34")
            || d.starts_with("37")
            || d.starts_with("30")
            || d.starts_with("36")
            || d.starts_with("38"))
}

fn is_valid_pci_digit_count(value: &str) -> bool {
    (12..=19).contains(&ascii_digits(value).len())
}

static VIN_RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^[A-HJ-NPR-Z0-9]{17}$").unwrap());
static PLATE_RE_1: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^[A-Z]{2,3}\d{4,6}$").unwrap());
static PLATE_RE_2: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^[A-Z]{1,6}\d{4,12}$").unwrap());

fn strip_vin_prefix(mut s: String) -> String {
    for prefix in ["VIN", "CHASSISNR", "CHASSIS", "STELNUMMER"] {
        if s.starts_with(prefix) {
            s.drain(..prefix.len());
            break;
        }
    }
    s
}

fn is_valid_vin(value: &str) -> bool {
    let c = strip_vin_prefix(compact(value).to_ascii_uppercase());
    c.len() == 17
        && c.chars().any(|ch| ch.is_ascii_alphabetic())
        && c.chars().any(|ch| ch.is_ascii_digit())
        && VIN_RE.is_match(&c).unwrap_or(false)
}

fn is_valid_plate_or_vin(value: &str) -> bool {
    let c = strip_vin_prefix(compact(value).to_ascii_uppercase());
    let digit_count = ascii_digits(&c).len();
    is_valid_vin(&c)
        || PLATE_RE_1.is_match(&c).unwrap_or(false)
        || PLATE_RE_2.is_match(&c).unwrap_or(false)
        || (c.chars().all(|ch| ch.is_ascii_alphanumeric())
            && (6..=18).contains(&c.len())
            && digit_count >= 2
            && c.chars().any(|ch| ch.is_ascii_alphabetic()))
}

// ── Compiled Rules ──────────────────────────────────────────────────

struct Rule {
    label: PiiLabel,
    pattern: Regex,
    validator: Option<fn(&str) -> bool>,
    priority: u32,
}

// `Rule` is `Send + Sync` because `fancy_regex::Regex` is `Send + Sync`
// and `fn` pointers are inherently thread-safe.
unsafe impl Send for Rule {}
unsafe impl Sync for Rule {}

fn compile(pattern: &str, ignore_case: bool) -> Regex {
    let flags = if ignore_case { "(?ix)" } else { "(?x)" };
    Regex::new(&format!("{flags}{pattern}"))
        .unwrap_or_else(|e| panic!("invalid regex pattern: {e}\n{pattern}"))
}

fn build_rules() -> Vec<Rule> {
    vec![
        // ── EMAIL ───────────────────────────────────────────────
        Rule {
            label: PiiLabel::EmailAddress,
            pattern: compile(
                r#"
                (?P<value>
                    \b
                    [a-z0-9.!#$%&'*+/=?^_`{|}~-]+
                    @
                    (?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+
                    [a-z]{2,63}
                    \b
                )
                "#,
                true,
            ),
            validator: None,
            priority: 10,
        },
        // ── IP ADDRESS ──────────────────────────────────────────
        Rule {
            label: PiiLabel::IpAddress,
            pattern: compile(
                r#"
                (?P<value>
                    \b(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)(?:\.(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)){3}\b
                    |
                    (?<![:\w])[a-f0-9:]*:[a-f0-9:]+(?![:\w])
                )
                "#,
                true,
            ),
            validator: Some(is_valid_ip),
            priority: 20,
        },
        // ── PCI (payment card) ──────────────────────────────────
        Rule {
            label: PiiLabel::Pci,
            pattern: compile(r"(?P<value>\b(?:\d[ -]*?){13,19}\b)", true),
            validator: Some(is_valid_pci_luhn),
            priority: 30,
        },
        Rule {
            label: PiiLabel::Pci,
            pattern: compile(
                r#"
                (?P<value>
                    \b(?:\d{4}[ -]?){3}\d{4}\b
                )
                "#,
                true,
            ),
            validator: None,
            priority: 31,
        },
        Rule {
            label: PiiLabel::Pci,
            pattern: compile(
                r#"
                (?P<value>
                    \b(?:\d{4}[ -]?\d{6}[ -]?\d{5})\b
                )
                "#,
                true,
            ),
            validator: None,
            priority: 31,
        },
        Rule {
            label: PiiLabel::Pci,
            pattern: compile(
                r#"
                \b(?:
                    card(?:\s+number)?
                    |credit\s+card(?:\s+number)?
                    |debit\s+card(?:\s+number)?
                    |visa(?:-kort)?(?:\s*|-)?(?:nummer|number)?
                    |mastercard(?:\s*|-)?(?:nummer|number)?
                    |kreditkort(?:nummer(?:et)?)?
                    |kortnummer(?:et)?
                    |kortbetaling
                    |kortoplysninger
                    |kortnumre
                    |kort(?:\s+med\s+nummer(?:et)?)?
                    |kortet
                    |betalingskort(?:nummer(?:et)?)?
                    |dankort(?:nummer(?:et)?)?
                    |bankkort(?:nummer(?:et)?)?
                )\b
                \s*[:\#\-]?\s*
                (?P<value>(?:\d[ -]*?){12,19})
                "#,
                true,
            ),
            validator: Some(is_valid_pci_digit_count),
            priority: 32,
        },
        Rule {
            label: PiiLabel::Pci,
            pattern: compile(
                r#"
                \b(?:
                    card|credit\s+card|debit\s+card|visa(?:-kort)?|mastercard|kreditkort|kort(?:nummer|betaling|oplysninger|numre)?|kortet|betalingskort|dankort|bankkort
                )\b
                [^\n\r\d]{0,40}
                (?P<value>(?:\d[ -]*?){12,19})
                (?=$|[^A-Za-z0-9])
                "#,
                true,
            ),
            validator: Some(is_valid_pci_digit_count),
            priority: 34,
        },
        Rule {
            label: PiiLabel::Pci,
            pattern: compile(
                r#"
                \b(?:cvv|cvc)\b\s*[:\#\-]?\s*(?P<value>\d{3,4})\b
                "#,
                true,
            ),
            validator: None,
            priority: 35,
        },
        Rule {
            label: PiiLabel::Pci,
            pattern: compile(
                r#"
                \b(?:udløbsdato|expiry|expiration\s+date|expires?)\b\s*[:\#\-]?\s*(?P<value>(?:0[1-9]|1[0-2])/[0-9]{2,4})\b
                "#,
                true,
            ),
            validator: None,
            priority: 36,
        },
        // ── BANK ACCOUNT ────────────────────────────────────────
        Rule {
            label: PiiLabel::BankAccount,
            pattern: compile(
                r"(?P<value>\b[A-Z]{2}\d{2}(?:[ -]?[A-Z0-9]{4}){2,7}[A-Z0-9]{0,2}\b)",
                true,
            ),
            validator: Some(is_valid_iban),
            priority: 40,
        },
        Rule {
            label: PiiLabel::BankAccount,
            pattern: compile(
                r#"
                \b
                (?:account(?:\s+number)?|bank\s+account|acct\.?|account\s*no\.?|konto(?:nummer)?|kontonr\.?)
                \b\s*[:\#\-]?\s*
                (?P<value>[A-Z0-9][A-Z0-9\-/\ ]{5,30}\d)
                "#,
                true,
            ),
            validator: Some(is_valid_account_number),
            priority: 45,
        },
        Rule {
            label: PiiLabel::BankAccount,
            pattern: compile(
                r#"
                \b(?:bankkonto|kontonummer|kontooplysninger|konto\s+til\s+(?:overførsel|udbetaling|lønudbetaling|refusion)|bankforbindelse)\b
                [^\n\r]{0,30}?
                (?P<value>\d{4}[ -]?\d{7,10}|\d{7,10})
                (?=$|[^A-Za-z0-9])
                "#,
                true,
            ),
            validator: Some(is_valid_account_number),
            priority: 46,
        },
        Rule {
            label: PiiLabel::BankAccount,
            pattern: compile(
                r#"
                \b(?:reg(?:istrerings)?nr\.?\s*/?\s*konto(?:nr\.)?|reg\.nr\.?\s*/\s*kontonr\.?|bank\s*\(reg\.nr\./konto\))\b
                \s*[:\#\-]?\s*
                (?P<value>\d{4}[ -]?\d{7,10})
                (?=$|[^A-Za-z0-9])
                "#,
                true,
            ),
            validator: Some(is_valid_account_number),
            priority: 47,
        },
        Rule {
            label: PiiLabel::BankAccount,
            pattern: compile(
                r#"
                \breg(?:istrerings)?nr\.?\s*[:\#\-]?\s*(?P<value>\d{4})\b
                "#,
                true,
            ),
            validator: Some(is_valid_account_number),
            priority: 47,
        },
        Rule {
            label: PiiLabel::BankAccount,
            pattern: compile(
                r#"
                \bkontonr\.?\s*[:\#\-]?\s*(?P<value>\d{7,10}|\d{4}[ -]\d{7,10})\b
                "#,
                true,
            ),
            validator: Some(is_valid_account_number),
            priority: 47,
        },
        // ── SSN (CPR) ──────────────────────────────────────────
        Rule {
            label: PiiLabel::Ssn,
            pattern: compile(
                r"(?P<value>\b(?:0[1-9]|[12]\d|3[01])(?:0[1-9]|1[0-2])\d{2}[- ]?\d{4}\b)",
                true,
            ),
            validator: Some(is_valid_cpr),
            priority: 50,
        },
        Rule {
            label: PiiLabel::Ssn,
            pattern: compile(
                r#"
                \b(?:cpr(?:-?nummer|nr\.?)?|personnummer|ssn|social\s*security\s*number)\b
                [^\n\r\d]{0,40}?
                (?P<value>\d{6}[- ]?\d{4})
                (?=$|[^A-Za-z0-9])
                "#,
                true,
            ),
            validator: None,
            priority: 51,
        },
        // ── PHONE NUMBER ────────────────────────────────────────
        Rule {
            label: PiiLabel::PhoneNumber,
            pattern: compile(r"(?P<value>\b\d{8}\b)", true),
            validator: Some(is_valid_phone),
            priority: 60,
        },
        Rule {
            label: PiiLabel::PhoneNumber,
            pattern: compile(
                r#"
                \b(?:tlf|telefon|mobil|phone|telefonnummer|mobilnummer)\b
                [^\n\r\d]{0,20}?
                (?P<value>\+?[0-9][0-9\ \-]{6,15}[0-9])
                (?=$|[^A-Za-z0-9])
                "#,
                true,
            ),
            validator: None,
            priority: 62,
        },
        Rule {
            label: PiiLabel::PhoneNumber,
            pattern: compile(
                r"(?P<value>(?<!\w)(?:\+\d{1,3}[ .\-]*)?(?:\(?\d{2,4}\)?[ .\-]*){2,4}\d{2,4}(?!\w))",
                true,
            ),
            validator: Some(is_valid_phone),
            priority: 65,
        },
        // ── PASSPORT ────────────────────────────────────────────
        Rule {
            label: PiiLabel::Passport,
            pattern: compile(
                r#"
                \b(?:passport(?:\s+number|\s+no\.?)|passnummer|passnr\.?|pasnummer(?:et)?|pasnr\.?)\b
                \s*[:\#\-]?\s*
                (?P<value>[A-Z]{0,2}\d{6,9})
                (?=$|[^A-Za-z0-9])
                "#,
                true,
            ),
            validator: Some(is_valid_passport_simple),
            priority: 47,
        },
        Rule {
            label: PiiLabel::Passport,
            pattern: compile(
                r#"
                \b(?:passport(?:\s+(?:number|no\.?|id|backup|copy|ved\s+check-in))?|passportnummer|passportkopi|passnummer|passnr\.?|pas(?:nummer(?:et)?|nr\.?|oplysninger?|information)?|rejsepas)\b
                \s*[:\#\-]?\s*
                (?P<value>[A-Z]{0,3}\d{6,9})
                (?=$|[^A-Za-z0-9])
                "#,
                true,
            ),
            validator: Some(is_valid_passport),
            priority: 48,
        },
        Rule {
            label: PiiLabel::Passport,
            pattern: compile(
                r#"
                \b(?:passport(?:\s+(?:number|no\.?|id|backup|copy|ved\s+check-in))?|passportnummer|passportkopi|passnummer|passnr\.?|pas(?:nummer(?:et)?|nr\.?|oplysninger?|information)?|rejsepas)\b
                [^\n\r]{0,50}?
                (?P<value>[A-Z]{0,3}\d{6,9})
                (?=$|[^A-Za-z0-9])
                "#,
                true,
            ),
            validator: Some(is_valid_passport),
            priority: 49,
        },
        Rule {
            label: PiiLabel::Passport,
            pattern: compile(
                r#"
                \b(?:passport(?:\s+(?:number|no\.?|id))?|passnummer|pasnummer|pasnr\.?|rejsepas)\b
                [^\n\r\d]{0,40}?
                (?P<value>[A-Z0-9]{6,12})
                (?=$|[^A-Za-z0-9])
                "#,
                true,
            ),
            validator: None,
            priority: 49,
        },
        Rule {
            label: PiiLabel::Passport,
            pattern: compile(r"(?P<value>\b\d{9}\b)", true),
            validator: None,
            priority: 52,
        },
        // ── DRIVERS LICENSE ─────────────────────────────────────
        Rule {
            label: PiiLabel::DriversLicense,
            pattern: compile(
                r#"
                \b(?:driver'?s?\s+licen[sc]e|driving\s+licen[sc]e|licen[sc]e\s+number|driver\s+license(?:\s+id)?|førerkort(?:\s+nr\.?)?|kørekort(?:nummer(?:et)?|oplysninger)?|kørekort\s+nr\.?)\b
                \s*[:\#\-]?\s*
                (?P<value>[A-Z0-9]+(?:-[A-Z0-9]+)*)
                (?=$|[^A-Za-z0-9])
                "#,
                true,
            ),
            validator: Some(is_valid_drivers_license),
            priority: 80,
        },
        Rule {
            label: PiiLabel::DriversLicense,
            pattern: compile(
                r#"
                \b(?:driver'?s?\s+licen[sc]e|driving\s+licen[sc]e|licen[sc]e\s+number|driver\s+license(?:\s+id)?|førerkort(?:\s+nr\.?)?|kørekort(?:nummer(?:et)?|oplysninger)?|kørekort\s+nr\.?)\w*\b
                (?:\s+(?:er|var|som|blev|oplyst(?:e)?|registreret(?:\s+som)?|fremvist|fundet|nummer(?:et)?))?
                \s*[:\#\-]?\s*
                (?P<value>(?:[A-Z]{1,6}[- ]?)?\d{5,12}|[A-Z]{1,6}(?:[- ]?[A-Z0-9]{2,}){2,}|[A-Z0-9]{5,20})
                (?=$|[^A-Za-z0-9])
                "#,
                true,
            ),
            validator: Some(is_valid_drivers_license),
            priority: 82,
        },
        Rule {
            label: PiiLabel::DriversLicense,
            pattern: compile(r"(?P<value>\b[A-Z0-9]{10}\b)", true),
            validator: Some(is_valid_dl_10char),
            priority: 85,
        },
        // ── VEHICLE ID ──────────────────────────────────────────
        Rule {
            label: PiiLabel::VehicleId,
            pattern: compile(r"(?P<value>\b[A-HJ-NPR-Z0-9]{17}\b)", true),
            validator: Some(is_valid_vin),
            priority: 90,
        },
        Rule {
            label: PiiLabel::VehicleId,
            pattern: compile(
                r#"
                \b(?:license\s+plate|number\s+plate|registration(?:\s+number)?|vehicle\s+id|vin|nummerplade|registreringsnummer)\b
                \s*[:\#\-]?\s*
                (?P<value>(?:[A-Z]{2,3}(?:[ -]?\d){4,6})|(?:[A-Z]{1,6}[ -]?\d{4,12})|(?:[A-Z0-9]{8,20}))
                (?=$|[^A-Za-z0-9])
                "#,
                true,
            ),
            validator: Some(is_valid_plate_or_vin),
            priority: 95,
        },
        Rule {
            label: PiiLabel::VehicleId,
            pattern: compile(
                r#"
                \b(?:køretøj(?:et)?|firmabil(?:en|ens)?|udlejningsbil(?:en|ens)?|nummerplade(?:n)?|reg(?:istrerings)?nr\.?|registreringsnummer(?:\s+for\s+køretøjet)?|leasede\s+køretøjer|trailer\s+id|stelnummer|chassisnr\.?)\b
                (?:\s+(?:er|var|som|med|på|for|:|nr\.?|nummer(?:plade)?|registreret(?:\s+som)?))?
                \s*[:\#\-]?\s*
                (?P<value>(?:[A-Z]{2,3}(?:[ -]?\d){4,6})|(?:[A-Z]{1,8}[ -]?\d{4,12})|(?:[A-Z0-9]{8,20}))
                (?=$|[^A-Za-z0-9])
                "#,
                true,
            ),
            validator: Some(is_valid_plate_or_vin),
            priority: 96,
        },
        Rule {
            label: PiiLabel::VehicleId,
            pattern: compile(
                r#"
                \b(?:bilen\s+.*?registreringsnummer|bilens\s+registreringsnummer|bilnummerplade|firmabilernes\s+nummerplader|fragtbil|lastbilplade|nummerplade(?:n)?|registreringsnummer(?:\s*\(nummerplade\))?|stelnummer|chassisnr\.?)\b
                [^\n\r]{0,45}?
                (?P<value>(?:[A-Z]{2,3}(?:[ -]?\d){4,6})|(?:VIN)?[A-Z0-9]{8,20})
                (?=$|[^A-Za-z0-9])
                "#,
                true,
            ),
            validator: Some(is_valid_plate_or_vin),
            priority: 97,
        },
        Rule {
            label: PiiLabel::VehicleId,
            pattern: compile(
                r#"
                \b(?:bilen|bilens|firmabil(?:en|ens)?|fragtbil(?:en)?|lastbilen)\b
                [^\n\r]{0,50}?\b(?:registreringsnummer|nummerplade|plade|registreret\s+som)\b
                [^\n\r]{0,20}?
                (?P<value>(?:[A-Z]{2,3}(?:[ -]?\d){4,6})|(?:VIN)?[A-Z0-9]{8,20})
                (?=$|[^A-Za-z0-9])
                "#,
                true,
            ),
            validator: Some(is_valid_plate_or_vin),
            priority: 98,
        },
        // ── PRECISE LOCATION ────────────────────────────────────
        Rule {
            label: PiiLabel::PreciseLocation,
            pattern: compile(
                r#"
                (?P<value>
                    \b[+-]?(?:90(?:\.0+)?|[1-8]?\d\.\d{3,})\s*[,;/]\s*[+-]?(?:180(?:\.0+)?|1[0-7]\d(?:\.\d+)?|\d{1,2}\.\d{3,})\b
                )
                "#,
                true,
            ),
            validator: None,
            priority: 100,
        },
        Rule {
            label: PiiLabel::PreciseLocation,
            pattern: compile(
                r#"
                (?P<value>
                    \b(?:[0-8]?\d(?:\.\d+)?|90(?:\.0+)?)°?\s*[NS]\s*[,;/]\s*(?:[0-9]{1,3}(?:\.\d+)?|1[0-7]\d(?:\.\d+)?|180(?:\.0+)?)°?\s*[EW]\b
                )
                "#,
                true,
            ),
            validator: None,
            priority: 101,
        },
        Rule {
            label: PiiLabel::PreciseLocation,
            pattern: compile(
                r#"
                (?P<value>
                    \b\d{1,5}[A-Z]?\s+
                    (?:[A-ZÆØÅ][\w'.\-]+\s+){0,4}
                    (?:street|st\.?|road|rd\.?|avenue|ave\.?|boulevard|blvd\.?|lane|ln\.?|drive|dr\.?|way|court|ct\.?|place|pl\.?|terrace|ter\.?|parkway|pkwy\.?|highway|hwy\.?)
                    \b(?:,?\s+[A-ZÆØÅ][\w'.\-]+){0,3}(?:,?\s+[A-Z]{2}\s+\d{5}(?:-\d{4})?)?
                )
                "#,
                false,
            ),
            validator: None,
            priority: 105,
        },
        Rule {
            label: PiiLabel::PreciseLocation,
            pattern: compile(
                r#"
                (?P<value>
                    \b(?:[A-ZÆØÅ][A-Za-zÆØÅæøå'.\-]*\s+){1,5}
                    \d{1,5}[A-Z]?
                    (?:,\s*(?:st\.?|th\.?|tv\.?|mf\.?|\d+\.\s*(?:th\.?|tv\.?|mf\.?)))?
                    (?:,\s*|\s+)\d{4}\s+[A-ZÆØÅ][A-Za-zÆØÅæøå.\-]*(?:\s+(?:[A-ZÆØÅ][A-Za-zÆØÅæøå.\-]*|[A-ZÆØÅ])){0,3}
                )
                "#,
                false,
            ),
            validator: None,
            priority: 108,
        },
        Rule {
            label: PiiLabel::PreciseLocation,
            pattern: compile(
                r#"
                (?P<value>
                    \b(?:[A-ZÆØÅ][\w'.\-]+\s+){0,4}
                    (?:
                        [A-ZÆØÅ][\w'.\-]*(?:gade|vej|vejen|alle|allé|boulevard|bakken|draget|stræde|stien|sti|havn|bro|park|parken|have|haven|vænget|holm|hus|høj|højen|skov|skoven|kær|kæret|mark|marken|dal|dalen|toft|toften|lunden|lund|gård|gården|bjerg|bjerget)
                        |
                        (?:[A-ZÆØÅ][\w'.\-]+\s+)?(?:Plads|Torv|Street|St\.?|Road|Rd\.?|Avenue|Ave\.?|Lane|Ln\.?|Drive|Dr\.?|Way|Court|Ct\.?|Place|Pl\.?|Terrace|Ter\.?|Parkway|Pkwy\.?|Highway|Hwy\.?)
                    )
                    \s+\d{1,5}[A-Z]?
                    (?:,\s*\d{4}\s*[A-ZÆØÅ][A-Za-zÆØÅæøå\-]*(?:\s+(?:[A-ZÆØÅ][A-Za-zÆØÅæøå\-]*|[A-ZÆØÅ])){0,3})?
                )
                "#,
                false,
            ),
            validator: None,
            priority: 110,
        },
        Rule {
            label: PiiLabel::PreciseLocation,
            pattern: compile(
                r#"
                (?P<value>
                    \b(?:[A-ZÆØÅ][a-zæøåA-ZÆØÅ.\-]+\s+){1,4}\d{1,4}[a-z]?[A-Z]?,\s+\d{4}\s+[A-ZÆØÅ][a-zæøåA-ZÆØÅ.\-]+
                )
                "#,
                false,
            ),
            validator: None,
            priority: 111,
        },
        Rule {
            label: PiiLabel::PreciseLocation,
            pattern: compile(
                r#"
                (?P<value>
                    \bpostboks\s+\d{1,5}(?:,\s*|\s+)\d{4}\s+[A-ZÆØÅ][A-Za-zÆØÅæøå.\-]*(?:\s+(?:[A-ZÆØÅ][A-Za-zÆØÅæøå.\-]*|[A-ZÆØÅ])){0,3}
                )
                "#,
                true,
            ),
            validator: None,
            priority: 112,
        },
        Rule {
            label: PiiLabel::PreciseLocation,
            pattern: compile(
                r#"
                (?P<value>
                    \b(?:[A-ZÆØÅ][A-Za-zÆØÅæøå'.\-]*(?:gade|vej|vejen|allé|alle|boulevard|plads|torv|stræde|sti|havn|bro)|[A-ZÆØÅ][A-Za-zÆØÅæøå'.\-]+(?:\s+[A-ZÆØÅ][A-Za-zÆØÅæøå'.\-]+){0,2})
                    \s+og\s+
                    (?:[A-ZÆØÅ][A-Za-zÆØÅæøå'.\-]*(?:gade|vej|vejen|allé|alle|boulevard|plads|torv|stræde|sti|havn|bro)|[A-ZÆØÅ][A-Za-zÆØÅæøå'.\-]+(?:\s+[A-ZÆØÅ][A-Za-zÆØÅæøå'.\-]+){0,2})
                    ,\s*\d{4}\s+[A-ZÆØÅ][A-Za-zÆØÅæøå.\-]*(?:\s+(?:[A-ZÆØÅ][A-Za-zÆØÅæøå.\-]*|[A-ZÆØÅ])){0,3}
                )
                "#,
                false,
            ),
            validator: None,
            priority: 114,
        },
    ]
}

static RULES: LazyLock<Vec<Rule>> = LazyLock::new(build_rules);

// ── Core Detection ──────────────────────────────────────────────────

fn iter_rule_spans(text: &str, rule: &Rule) -> Vec<Span> {
    let mut spans = Vec::new();
    for caps_result in rule.pattern.captures_iter(text) {
        let caps = match caps_result {
            Ok(caps) => caps,
            Err(_) => continue,
        };

        let m = match caps.name("value").or_else(|| caps.get(0)) {
            Some(m) => m,
            None => continue,
        };

        let start = m.start();
        let end = m.end();
        let raw = &text[start..end];
        let candidate = raw.trim_matches(|c: char| " ,;:\n\t".contains(c));

        if candidate.is_empty() {
            continue;
        }

        if let Some(validator) = rule.validator {
            if !validator(candidate) {
                continue;
            }
        }

        // Adjust span to the trimmed candidate within the original slice.
        let offset = raw.find(candidate).unwrap_or(0);
        spans.push(Span {
            start: start + offset,
            end: start + offset + candidate.len(),
            label: rule.label,
        });
    }
    spans
}

fn select_non_overlapping(mut spans: Vec<(u32, Span)>) -> Vec<Span> {
    // Sort by: start position, then priority (lower = higher), then prefer longer spans.
    spans.sort_by(|a, b| {
        a.1.start
            .cmp(&b.1.start)
            .then(a.0.cmp(&b.0))
            .then((b.1.end - b.1.start).cmp(&(a.1.end - a.1.start)))
    });

    let mut selected: Vec<Span> = Vec::new();
    let mut seen: HashSet<(usize, usize, PiiLabel)> = HashSet::new();

    for (_, span) in spans {
        let key = (span.start, span.end, span.label);
        if seen.contains(&key) {
            continue;
        }
        if selected
            .iter()
            .any(|existing| span.start < existing.end && existing.start < span.end)
        {
            continue;
        }
        seen.insert(key);
        selected.push(span);
    }

    selected
}

// ── Public API ──────────────────────────────────────────────────────

/// Detect PII spans in `text` using compiled regex rules.
///
/// Returns non-overlapping [`Span`]s sorted by position. When two rules match
/// the same region, the one with the lower priority value (higher precedence)
/// wins. Pass `label_whitelist` to restrict detection to specific categories.
///
/// This function is **`Send + Sync`** safe — call it from any thread or via
/// `rayon::par_iter` over a batch of texts.
pub fn detect_pii(text: &str, label_whitelist: Option<&HashSet<PiiLabel>>) -> Vec<Span> {
    let active_labels: HashSet<PiiLabel> = match label_whitelist {
        Some(whitelist) => REGEX_COMPATIBLE_LABELS
            .iter()
            .copied()
            .filter(|l| whitelist.contains(l))
            .collect(),
        None => REGEX_COMPATIBLE_LABELS.iter().copied().collect(),
    };

    let mut prioritized: Vec<(u32, Span)> = Vec::new();

    for rule in RULES.iter() {
        if !active_labels.contains(&rule.label) {
            continue;
        }
        for span in iter_rule_spans(text, rule) {
            prioritized.push((rule.priority, span));
        }
    }

    select_non_overlapping(prioritized)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_email() {
        let spans = detect_pii("Send til user@example.com snarest", None);
        assert_eq!(spans.len(), 1);
        assert_eq!(spans[0].label, PiiLabel::EmailAddress);
        assert_eq!(
            &"Send til user@example.com snarest"[spans[0].start..spans[0].end],
            "user@example.com"
        );
    }

    #[test]
    fn detects_cpr() {
        let spans = detect_pii("CPR-nummer: 010190-1234", None);
        assert!(spans.iter().any(|s| s.label == PiiLabel::Ssn));
    }

    #[test]
    fn detects_iban() {
        let spans = detect_pii("Konto: DK50 0040 0440 1162 43", None);
        assert!(spans.iter().any(|s| s.label == PiiLabel::BankAccount));
    }

    #[test]
    fn detects_danish_address() {
        let spans = detect_pii("Bor på Vestergade 12, 1456 København K", None);
        assert!(spans.iter().any(|s| s.label == PiiLabel::PreciseLocation));
    }

    #[test]
    fn respects_whitelist() {
        let mut whitelist = HashSet::new();
        whitelist.insert(PiiLabel::EmailAddress);
        let text = "Email: a@b.com, CPR: 010190-1234";
        let spans = detect_pii(text, Some(&whitelist));
        assert!(spans.iter().all(|s| s.label == PiiLabel::EmailAddress));
    }

    #[test]
    fn no_overlapping_spans() {
        let text = "Kontakt mig på +45 12345678 eller test@mail.dk";
        let spans = detect_pii(text, None);
        for (i, a) in spans.iter().enumerate() {
            for b in spans.iter().skip(i + 1) {
                assert!(
                    a.end <= b.start || b.end <= a.start,
                    "overlapping spans: {:?} and {:?}",
                    a,
                    b
                );
            }
        }
    }

    #[test]
    fn is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<Span>();
        assert_send_sync::<PiiLabel>();
    }
}
