use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "fortrolig", about = "Regex-based PII detection service")]
pub struct Cli {
    #[command(subcommand)]
    pub command: Command,
}

#[derive(Subcommand)]
pub enum Command {
    /// Start the REST API server
    Serve {
        /// Host to bind to
        #[arg(long, default_value = "127.0.0.1")]
        host: String,

        /// Port to listen on
        #[arg(short, long, default_value_t = 3000)]
        port: u16,
    },

    /// Run a demo showing PII detection on example Danish text
    Example,
}
