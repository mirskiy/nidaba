# -*- coding: utf-8 -*-
import re
from lxml import etree
from kitchen.text.converters import to_unicode, to_bytes
from PIL import Image, ImageDraw


def extract_hocr_tokens(hocr_file):
    """
    Extracts all the nonempty words in an hOCR file and returns them
    as a list.
    """
    words = []
    context = etree.iterparse(hocr_file, events=('end',), tag='span', html=True)
    for event, element in context:
        # Strip extraneous newlines generated by the ocr_line span tags.
        if element.text is not None:
            word = to_unicode(element.text.rstrip())
        if len(word) > 0:
            words.append(word)
        element.clear()
        while element.getprevious() is not None:
            del element.getparent()[0]
    del context
    return words



def extract_bboxes(hocr_file):
    """
    Extracts a list of bboxes as 4-tuples, in the same order that they
    appear in the hocr file.
    """
    context = etree.parse(hocr_file)
    elements_with_bbox = context.xpath(u'//@title')

    pattern = r'.*(bbox{1} [0-9]+ [0-9]+ [0-9]+ [0-9]+)'
    bboxes = []
    for e in elements_with_bbox:
        match = re.match(pattern, str(e))
        bbox = tuple(map(int, match.groups()[0][5:].split(u' ')))
        bboxes.append(bbox)
    return bboxes

def extract_bboxes_by_class(hocrfile, classes):
    """
    Extracts bboxes of the specified tags. Returns a dictionary which
    maps tag names to lists of bboxs. These lists are ordered as the are
    in the hocr document.
    """
    context = etree.parse(hocrfile)
    results = {hocrclass:[] for hocrclass in classes}
    pattern = r'.*(bbox{1} [0-9]+ [0-9]+ [0-9]+ [0-9]+)'

    for c in classes:
        xpath = u"//*[@class='%s' and @title]" % c
        for element in context.xpath(xpath):
            match = re.match(pattern, element.attrib[u'title'])
            bbox = tuple(map(int, match.groups()[0][5:].split(u' ')))
            results[c].append(bbox)

    return results

def previewbboxs(imgfile, hocrfile, color='blue'):
    """
    Display a preview of the specified image with the bboxes from the
    hocr file drawn on it.
    """
    opened = Image.open(imgfile)
    draw = ImageDraw.Draw(opened)
    for bbox in extract_bboxes(hocrfile):
        draw.rectangle(((bbox[0], bbox[1]),(bbox[2], bbox[3])), outline=color)

    opened.show()


# if __name__ == '__main__':
