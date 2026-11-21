from .app import main

# Guard needed: separation uses multiprocessing "spawn", which re-imports __main__.
if __name__ == "__main__":
    main()
