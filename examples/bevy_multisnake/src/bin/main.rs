use bevy_multisnake::run;
use clap::Parser;

#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
struct Snake {
    #[clap(long, value_parser)]
    agent_path: Option<String>,

    #[clap(long, value_parser)]
    agent2_path: Option<String>,

    #[clap(long, action)]
    easy_mode: bool,
}

fn main() {
    let args = Snake::parse();
    run(args.agent_path, args.agent2_path, args.easy_mode);
}
