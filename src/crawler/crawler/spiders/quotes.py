
import scrapy

from ..utils.misc_utils import save_json


QUOTES_FNAME = "quotes.json"
QUOTES_URLS = [
        "https://quotes.toscrape.com/page/1/",
        "https://quotes.toscrape.com/page/2/",
    ]



# class QuotesSpider(scrapy.Spider):
#     name = "quotes"
#     start_urls = [
#         "https://quotes.toscrape.com/page/1/",
#         "https://quotes.toscrape.com/page/2/",
#     ]
#
#     def parse(self, response):
#         page = response.url.split("/")[-2]
#         filename = f"quotes-{page}.html"
#         Path(filename).write_bytes(response.body)

class QuotesSpider(scrapy.Spider):
    name = "quotes"
    start_urls = QUOTES_URLS

    # def parse(self, response):
    #     for quote in response.css("div.quote"):
    #         yield {
    #             "text": quote.css("span.text::text").get(),
    #             "author": quote.css("small.author::text").get(),
    #             "tags": quote.css("div.tags a.tag::text").getall(),
    #         }

    def parse(self, response):
        info = []
        for quote in response.css("div.quote"):
            entry = {
                "text": quote.css("span.text::text").get(),
                "author": quote.css("small.author::text").get(),
                "tags": quote.css("div.tags a.tag::text").getall(),
                "href": quote.css("div.tags a.tag::attr(href)").getall()
            }
            info.append(entry)
        save_json(QUOTES_FNAME, info)
