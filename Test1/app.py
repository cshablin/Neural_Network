from flask import Flask

from Test1.service import SomeService

service = SomeService()


def init_routes():
    @app.route('/')
    def get_top_10_rows():
        # return 'hello'
        return service.get_top_10_rows('Michael')


if __name__ == '__main__':
    app = Flask(__name__)
    init_routes()
    app.run(host="127.0.0.1", port=int(5500))

