import asyncio
import os
from pyppeteer import launch


class Webscreenshooter():
    def __init__(self, size):
        self.size = size
    def gen_output_path(self, from_url, output_dir):
        base_name = from_url.replace('.html', '').split('\\')[-1]
        file_name = "{}.png".format(base_name)
        output_path = os.path.join(output_dir, file_name)
        return output_path, file_name

    async def screenshot(self, url, output_path):
        browser = await launch(args=['--no-sandbox'])
        page = await browser.newPage()
        await page.setViewport({'width': self.size[0], 'height': self.size[1]})
        await page.goto(url)
        await page.screenshot({'path': output_path})
        await browser.close()

    def take_screenshot(self, url, output_path):
        asyncio.get_event_loop().run_until_complete(self.screenshot(url, output_path))
        print('web screenshot complete')


def webscreenshoot(urls: list, output_dir, size=(2400, 3000)):
    output_file_list = []
    ws = Webscreenshooter(size)
    for url in urls:
        output_path, file_name = ws.gen_output_path(url, output_dir)
        ws.take_screenshot(url, output_path)
        output_file_list.append(output_path)
    return output_file_list



if __name__ == '__main__':
    ws = Webscreenshooter()
    url = 'file:///E:/projects/NTUST/webimg-to-code/dataset/dataset2/origin_html/2.html'
    output_dir = 'E:\\projects\\NTUST\\webimg-to-code\\datasetCode\\data_transform\\assest'

    output_path, file_name = ws.gen_output_path(url, output_dir)
    ws.take_screenshot(url, output_path)
    print(output_path, file_name)