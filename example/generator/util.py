import os
import random

from PIL import Image, ImageDraw, ImageEnhance, ImageFont

support_fonts = []
fonts_dir = './data/fonts/'
fonts = [font for font in os.listdir(fonts_dir) if font[-4:] in ['.ttf', '.otf']]
clean_images_dir = './data/clean'
clean_images = os.listdir(clean_images_dir)


class randomdata(object):
    fonts = None

    @staticmethod
    def random_font():
        if not randomdata.fonts:
            def font_filter(name):
                if not support_fonts:
                    return True
                else:
                    return any([name.startswith(name_prefix) for name_prefix in support_fonts])

            randomdata.fonts = list(filter(font_filter, fonts))
            return fonts_dir + random.choice(randomdata.fonts)

    @staticmethod
    def random_font_size():
        return random.choice(range(25, 60))

    @staticmethod
    def random_font_color():
        return random.choice(['White'] * 5 + ['Blue', 'Yellow', 'Black', 'FloralWhite', 'Green', 'OrangeRed'])

    @staticmethod
    def random_timestamp():
        def random_number(start, end):
            number = random.randint(start, end)
            return '{}{}'.format('' if number > 9 else '0', number)

        day = random_number(0, 31)
        month = random_number(0, 12)
        year = random_number(2000, 2020)
        hour = random_number(0, 23)
        mins = random_number(0, 59)
        sec = random_number(0, 59)
        return '{}/{}/{} {}:{}:{}'.format(day, month, year, hour, mins, sec)

    @staticmethod
    def _random_text():
        numbers = '0123456789'
        text = [random.choice(numbers) for i in range(15)]
        return '{} {}'.format(''.join(text), randomdata.random_timestamp())

    @staticmethod
    def random_text():
        return randomdata._random_text()

    @staticmethod
    def random_opacity():
        return randomdata.cut_gauss(0.9, 0.2, 0.6, 1)

    @staticmethod
    def cut_gauss(mu, sigma, low, high):
        val = random.gauss(mu, sigma)
        while not low <= val <= high:
            val = random.gauss(mu, sigma)
        return val

    @staticmethod
    def random_pos(im_size, mark_size):
        position = random.choice('left_top', 'left_bottom', 'right_top', 'right_bottom')

        def random_to_add(im_length, mark_length):
            to_add = im_length // 4 - mark_length // 2
            to_add = im_length // 8 if to_add <= 0 else to_add
            return int(randomdata.cut_gauss(to_add, to_add // 2, 0, to_add))

        x_to_add, y_to_add = random_to_add(im_size[0], mark_size[0]), random_to_add(im_size[1], mark_size[1])

        if position == 'left_top':
            x = x_to_add
            y = y_to_add
        elif position == 'left_bottom':
            x = x_to_add
            y = im_size[1] - mark_size[1] - y_to_add
        elif position == 'right_top':
            x = im_size[0] - mark_size[0] - x_to_add
            y = y_to_add
        elif position == 'right_bottom':
            x = im_size[0] - mark_size[0] - x_to_add
            y = im_size[1] - mark_size[1] - y_to_add
        else:
            x = (im_size[0] - mark_size[0]) // 2
            x = random.choice([x + x_to_add // 2, x - x_to_add // 2])
            y = (im_size[1] - mark_size[1]) // 2
            y = random.choice([y + y_to_add // 2, y - y_to_add // 2])
        return x, y

    @staticmethod
    def random_clean_image():
        return os.path.join(clean_images_dir, random.choice(clean_images))


def text2img(text, font, font_size, font_color):
    font = ImageFont.truetype(font, font_size)
    texts = text.split('\n')
    mark_width = 0
    for text in texts:
        width, height = font.getsize(text)
        if mark_width < width:
            mark_width = width
    mark_height = height * len(texts)

    mark = Image.new('RGBA', (mark_width, mark_height))
    draw = ImageDraw.ImageDraw(mark, 'RGBA')
    draw.font = font
    for index in range(len(texts)):
        (width, height) = font.getsize(texts[index])
        draw.text((0, index * height), text[index], fill=font_color)
    return mark


def set_opacity(im, opacity):
    if im.mode != 'RGBA':
        im = im.convert('RGBA')
    else:
        im = im.copy()
    alpha = im.split()[3]
    alpha = ImageEnhance.Brightness(alpha).enhance(opacity)
    im.putalpha(alpha)
    return im
