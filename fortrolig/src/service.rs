use std::collections::HashSet;
use std::net::SocketAddr;

use axum::{Json, Router, routing::post};
use serde::{Deserialize, Serialize};
use utoipa::OpenApi;
use utoipa::ToSchema;
use utoipa_redoc::{Redoc, Servable as RedocServable};
use utoipa_scalar::{Scalar, Servable as ScalarServable};
use utoipa_swagger_ui::SwaggerUi;

use crate::regex_model::{PiiLabel, REGEX_COMPATIBLE_LABELS, detect_pii};

// ── Request / Response types ────────────────────────────────────────

#[derive(Deserialize, ToSchema)]
struct DetectRequest {
    /// The text to scan for PII.
    text: String,
    /// Optional list of PII label names to restrict detection to.
    /// If omitted, all supported labels are used.
    #[schema(example = json!(["EMAIL_ADDRESS", "SSN"]))]
    labels: Option<Vec<String>>,
}

#[derive(Serialize, ToSchema)]
struct SpanResponse {
    /// Byte offset where the PII starts (UTF-8).
    start: usize,
    /// Byte offset where the PII ends (UTF-8, exclusive).
    end: usize,
    /// The PII category label.
    #[schema(example = "EMAIL_ADDRESS")]
    label: &'static str,
    /// The matched text.
    #[schema(example = "user@example.com")]
    text: String,
}

#[derive(Serialize, ToSchema)]
struct DetectResponse {
    /// Detected PII spans, sorted by position.
    spans: Vec<SpanResponse>,
}

#[derive(Serialize, ToSchema)]
struct LabelsResponse {
    /// All PII label names supported by the regex engine.
    labels: Vec<&'static str>,
}

// ── Handlers ────────────────────────────────────────────────────────

#[utoipa::path(
    post,
    path = "/detect",
    request_body = DetectRequest,
    responses(
        (status = 200, description = "Detected PII spans", body = DetectResponse),
    ),
    tag = "PII Detection",
    summary = "Detect PII in text",
    description = "Scans the provided text for personally identifiable information using regex rules. Returns non-overlapping spans sorted by position.",
)]
async fn detect_handler(Json(req): Json<DetectRequest>) -> Json<DetectResponse> {
    let whitelist = req.labels.as_ref().map(|labels| {
        labels
            .iter()
            .filter_map(|l| PiiLabel::from_label_str(l))
            .collect::<HashSet<_>>()
    });

    let spans = detect_pii(&req.text, whitelist.as_ref());

    let response_spans = spans
        .into_iter()
        .map(|span| SpanResponse {
            text: req.text[span.start..span.end].to_string(),
            start: span.start,
            end: span.end,
            label: span.label.as_str(),
        })
        .collect();

    Json(DetectResponse {
        spans: response_spans,
    })
}

#[utoipa::path(
    get,
    path = "/labels",
    responses(
        (status = 200, description = "Supported PII labels", body = LabelsResponse),
    ),
    tag = "PII Detection",
    summary = "List supported PII labels",
    description = "Returns the list of all PII category labels that the regex engine can detect.",
)]
async fn labels_handler() -> Json<LabelsResponse> {
    Json(LabelsResponse {
        labels: REGEX_COMPATIBLE_LABELS.iter().map(|l| l.as_str()).collect(),
    })
}

// ── OpenAPI ─────────────────────────────────────────────────────────

#[derive(OpenApi)]
#[openapi(
    info(
        title = "Fortrolig",
        description = "Fast regex-based PII detection service for Danish and international text",
        version = "0.1.0",
    ),
    paths(detect_handler, labels_handler),
    components(schemas(DetectRequest, DetectResponse, SpanResponse, LabelsResponse)),
    tags(
        (name = "PII Detection", description = "Endpoints for detecting and listing PII categories"),
    ),
)]
struct ApiDoc;

// ── Server ──────────────────────────────────────────────────────────

fn app() -> Router {
    Router::new()
        .route("/detect", post(detect_handler))
        .route("/labels", axum::routing::get(labels_handler))
        .merge(Redoc::with_url("/redoc", ApiDoc::openapi()))
        .merge(Scalar::with_url("/scalar", ApiDoc::openapi()))
        .merge(
            SwaggerUi::new("/docs")
                .url("/api-docs/openapi.json", ApiDoc::openapi())
                .config(
                    utoipa_swagger_ui::Config::from("/api-docs/openapi.json")
                        .doc_expansion("list")
                        .deep_linking(true)
                        .default_models_expand_depth(1)
                        .display_request_duration(true),
                ),
        )
}

pub async fn run(host: &str, port: u16) {
    let app = app();
    let addr: SocketAddr = format!("{host}:{port}")
        .parse()
        .expect("invalid host:port address");
    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .expect("failed to bind");
    println!("Listening on http://{addr}");
    println!("Swagger UI: http://{addr}/docs");
    println!("Scalar:     http://{addr}/scalar");
    println!("Redoc:      http://{addr}/redoc");
    axum::serve(listener, app).await.expect("server error");
}
