import multiprocessing
import time
import updater


def main():
    multiprocessing.Process(target=updater.listen_forever_sync, daemon=True).start()
    while True:
        print("Hello World")
        time.sleep(5)


if __name__ == "__main__":
    main()
