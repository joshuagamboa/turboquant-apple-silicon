fn main() {
    print!("\x1b[?1049h"); // alt screen
    print!("\x1b[?7l");    // disable auto-wrap
    println!("Look ma, no wrap on the last col!");
    std::thread::sleep(std::time::Duration::from_millis(100));
    print!("\x1b[?7h");    // enable auto-wrap
    print!("\x1b[?1049l"); // prev screen
}
