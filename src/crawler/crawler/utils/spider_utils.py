
from scrapy.utils.reactor import install_reactor
install_reactor('twisted.internet.asyncioreactor.AsyncioSelectorReactor')

from scrapy.crawler import CrawlerProcess
from scrapy.utils.project import get_project_settings


def run_crawler(spider_name: str):
    """Run isolated spider and restart reactor to run another spider afterwards."""
    process = CrawlerProcess(get_project_settings())
    process.crawl(spider_name)
    process.start()

    import sys
    del sys.modules['twisted.internet.reactor']