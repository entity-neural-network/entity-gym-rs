use bevy_snake_enn::run;
use clap::Parser;

#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
struct Snake {
    #[clap(long, value_parser)]
    agent_path: Option<String>,
}

fn main() {
    let args = Snake::parse();
    run(args.agent_path);
}
