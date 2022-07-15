# questions

- run bevy env in headless mode:
  - disable graphics. also not spawn graphics-related components?
  - FixedTimestep: manually convert code to run in logical time? some way to automatically convert FixedTimestep to logical time and run as fast as possible?


```
thread '<unnamed>' panicked at 'Initializing the event loop outside of the main thread is a significant cross-platform compatibility hazard. If you really, absolutely need to create an EventLoop on a different thread, please use the `EventLoopExtUnix::new_any_thread` function.', /home/clemens/.cargo/registry/src/github.com-1ecc6299db9ec823/winit-0.26.1/src/platform_impl/linux/mod.rs:758:9
```
