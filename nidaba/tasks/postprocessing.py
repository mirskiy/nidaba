# -*- coding: utf-8 -*-
"""
nidaba.tasks.postprocessing
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Various postprocessing tasks that operate upon recognized texts.

"""

from __future__ import unicode_literals, print_function, absolute_import

import itertools

from nidaba import storage
from nidaba import merge_hocr
from nidaba import lex
from nidaba.celery import app
from nidaba.tei import TEIFacsimile
from nidaba.config import nidaba_cfg
from nidaba.algorithms import median
from nidaba.tasks.helper import NidabaTask

from celery.utils.log import get_task_logger

logger = get_task_logger(__name__)


@app.task(base=NidabaTask, name=u'nidaba.postprocessing.spell_check', 
          arg_values={'language': nidaba_cfg['lang_dicts'].keys(),
                      'filter_punctuation': [True, False]})
def spell_check(doc, method=u'spell_check', language=u'',
                filter_punctuation=False):
    """
    Adds spelling suggestions to an TEI XML document.

    Alternative spellings for each segment will be included in a choice
    tagcontaining a series of corr tags with the original segment appearing
    beneath a sic element.  Correct words, i.e. words appearing verbatim in the
    dictionary, are left untouched.

    Args:
        doc (unicode, unicode): The input document tuple.
        method (unicode): The suffix string appended to the output file.
        language (unicode): Identifier defined in the nidaba configuration as a
                            valid dictionary.
        filter_punctuation (bool): Switch to filter punctuation inside
                                   ``seg``
    Returns:
        (unicode, unicode): Storage tuple of the output document
    """
    input_path = storage.get_abs_path(*doc)
    output_path = storage.insert_suffix(input_path, method, language,
                                        unicode(filter_punctuation))
    dictionary = storage.get_abs_path(*nidaba_cfg['lang_dicts'][language]['dictionary'])
    del_dictionary = storage.get_abs_path(*nidaba_cfg['lang_dicts'][language]['deletion_dictionary'])
    with storage.StorageFile(*doc) as fp:
        logger.debug('Reading TEI ({})'.format(fp.abs_path))
        tei = TEIFacsimile()
        tei.read(fp)
        logger.debug('Performing spell check')
        ret = lex.tei_spellcheck(tei, dictionary, del_dictionary,
                                 filter_punctuation)
    with storage.StorageFile(*storage.get_storage_path(output_path), mode='wb') as fp:
        logger.debug('Writing TEI ({})'.format(fp.abs_path))
        ret.write(fp)
    return storage.get_storage_path(output_path)


@app.task(base=NidabaTask, name=u'nidaba.postprocessing.blend', type='merge')
def blend(doc, method=u'blend'):
    """
    Blends multiple hOCR files using the algorithm I cooked up in between
    thesis procrastination sessions. It is language independent and
    parameterless.

    Args:
        doc [(id, path), ...]: A list of storage module tuples that will be
        merged into a single output document.
        method (unicode): The suffix string appended to the output file.

    Returns:
        (unicode, unicode): Storage tuple of the output document
    """
    # create the output document path from the first input document
    input_path = storage.get_abs_path(*doc[0])
    output_path = storage.insert_suffix(input_path, method)
    teis = []
    output = None
    for i in doc:
        with storage.StorageFile(*i) as fp:
            logger.debug('Reading TEI ({})'.format(fp.abs_path))
            tei = TEIFacsimile()
            tei.read(fp)
            teis.append(tei.doc.iter(tei.tei_ns + 'line'))
            # built skeleton of output file
            if output is None:
                fp.seek(0)
                output = TEIFacsimile()
                output.read(fp)
                output.clear_segments()
                output.add_respstmt('merge', 'confidence-weighted')
    for lines in itertools.izip(*teis):
        recs = []
        line_id = lines[0].get(output.xml_ns + 'id')
        output.scope_line(line_id)
        logger.debug('Calculating median for line {}'.format(line_id))
        for line in lines:
            certs = []
            text = u''
            for g in line.iter(output.tei_ns + 'g'):
                text += ''.join(g.itertext())
                # no standoff notation for certainties
                certs.append(float(line.xpath(".//*[local-name()='certainty' and @target=$tag]", 
                                              tag='#' + g.get(output.xml_ns +'id'))[0].get('degree')))
            recs.append(median.ocr_record(text, len(text) * [None], certs))
        m = median.approximate_median(recs, median.confidence_weighted_edit_distance)
        logger.debug('Calculated median: {}'.format(m))
        output.add_graphemes(x for x in m.prediction)
    with storage.StorageFile(*storage.get_storage_path(output_path), mode='wb') as fp:
        logger.debug('Writing TEI ({})'.format(fp.abs_path))
        output.write(fp)
    return storage.get_storage_path(output_path)
