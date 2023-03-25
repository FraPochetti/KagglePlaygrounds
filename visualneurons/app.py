import os
import tornado.web

from .server import StyleTransferSocket

from tornado.options import define, options


class Application(tornado.web.Application):
    def __init__(self):
        handlers = [("/styletransfer", StyleTransferSocket)]
        super().__init__(
            handlers,
            cookie_secret="__TODO:_GENERATE_YOUR_OWN_RANDOM_VALUE_HERE__",
            template_path=os.path.join(os.path.dirname(__file__), "templates"),
            static_path=os.path.join(os.path.dirname(__file__), "static"),
            xsrf_cookies=True,
        )


def main():
    define("port", default=8000, help="Run on the given port.")
    define("address", default="localhost", help="Run on the given address.")
    tornado.options.parse_command_line()

    app = Application()
    app.listen(options.port, options.address)
    tornado.ioloop.IOLoop.current().start()


if __name__ == "__main__":
    main()
