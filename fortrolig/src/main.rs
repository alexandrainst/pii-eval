mod cli;
mod regex_model;
mod service;

use clap::Parser;
use cli::{Cli, Command};
use regex_model::detect_pii;

#[tokio::main]
async fn main() {
    let cli = Cli::parse();
    match cli.command {
        Command::Serve { host, port } => {
            service::run(&host, port).await;
        }
        Command::Example => {
            let text = "Mit CPR-nummer er 01-01-90-1234 og min email er test@example.com. Min bil har nummerplade: DD 12 312";
            println!("Input: {text}");
            for span in detect_pii(text, None) {
                println!(
                    "[{}] ({}-{}): {}",
                    span.label, span.start, span.end, &text[span.start..span.end],
                );
            }
        }
    }
}
