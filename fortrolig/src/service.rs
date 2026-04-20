use std::collections::HashSet;
use std::net::SocketAddr;

use axum::{Json, Router, routing::post};
use serde::{Deserialize, Serialize};

use crate::regex_model::{PiiLabel, REGEX_COMPATIBLE_LABELS, detect_pii};

// ── Request / Response types ────────────────────────────────────────

#[derive(Deserialize)]
struct DetectRequest {
    text: String,
    labels: Option<Vec<String>>,
}

#[derive(Serialize)]
struct SpanResponse {
    start: usize,
    end: usize,
    label: &'static str,
    text: String,
}

#[derive(Serialize)]
struct DetectResponse {
    spans: Vec<SpanResponse>,
}

#[derive(Serialize)]
struct LabelsResponse {
    labels: Vec<&'static str>,
}

// ── Handlers ────────────────────────────────────────────────────────

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

async fn labels_handler() -> Json<LabelsResponse> {
    Json(LabelsResponse {
        labels: REGEX_COMPATIBLE_LABELS.iter().map(|l| l.as_str()).collect(),
    })
}

// ── Server ──────────────────────────────────────────────────────────

fn app() -> Router {
    Router::new()
        .route("/detect", post(detect_handler))
        .route("/labels", axum::routing::get(labels_handler))
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
    axum::serve(listener, app).await.expect("server error");
}
