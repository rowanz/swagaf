"""
adapted from Allennlp because their version doesn't seem to work 😢😢😢
"""
from typing import Dict, Any, Iterable
import argparse
import logging

from allennlp.commands.subcommand import Subcommand
from allennlp.common.util import prepare_environment, import_submodules
from allennlp.common.tqdm import Tqdm
from allennlp.data import Instance
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.iterators import DataIterator
from allennlp.models.archival import load_archive
from allennlp.models.model import Model
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name






if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="predict")

    parser.add_argument('archive_file', type=str, help='path to an archived trained model')

    parser.add_argument('input_file', type=str, help='path to input file')

    cuda_device = parser.add_mutually_exclusive_group(required=False)
    cuda_device.add_argument('--cuda-device',
                             type=int,
                             default=0,
                             help='id of GPU to use (if any)')

    parser.add_argument('-o', '--overrides',
                           type=str,
                           default="",
                           help='a HOCON structure used to override the experiment configuration')

    parser.add_argument('--include-package',
                           type=str,
                           action='append',
                           default=[],
                           help='additional packages to include')

    parser.add_argument('--output-file', type=str, help='path to output file')
    args = parser.parse_args()

    # Disable some of the more verbose logging statements
    logging.getLogger('allennlp.common.params').disabled = True
    logging.getLogger('allennlp.nn.initializers').disabled = True
    logging.getLogger('allennlp.modules.token_embedders.embedding').setLevel(logging.INFO)

    # Import any additional modules needed (to register custom classes)
    for package_name in args.include_package:
        import_submodules(package_name)

    # Load from archive
    import ipdb
    ipdb.set_trace()
    archive = load_archive(args.archive_file, args.cuda_device, args.overrides)
    config = archive.config
    prepare_environment(config)
    model = archive.model
    model.eval()

    # Load the evaluation data

    # Try to use the validation dataset reader if there is one - otherwise fall back
    # to the default dataset_reader used for both training and validation.
    validation_dataset_reader_params = config.pop('validation_dataset_reader', None)
    if validation_dataset_reader_params is not None:
        dataset_reader = DatasetReader.from_params(validation_dataset_reader_params)
    else:
        dataset_reader = DatasetReader.from_params(config.pop('dataset_reader'))
    evaluation_data_path = args.input_file
    logger.info("Reading evaluation data from %s", evaluation_data_path)
    instances = dataset_reader.read(evaluation_data_path)

    config['iterator']['type'] = 'basic'
    del config['iterator']['sorting_keys']
    data_iterator = DataIterator.from_params(config.pop("iterator"))
    data_iterator.index_with(model.vocab)

    cuda_device = args.cuda_device

    #### EVALUATION AQUI


    model.eval()
    iterator = data_iterator(instances, num_epochs=1, shuffle=False, cuda_device=cuda_device, for_training=False)
    logger.info("Iterating over dataset")
    generator_tqdm = Tqdm.tqdm(iterator, total=data_iterator.get_num_batches(instances))

    label_probs = []
    for batch in generator_tqdm:
        lol = model(**batch)
        label_probs.append(model(**batch)['label_probs'].data.cpu().numpy())
    label_probs = np.concatenate(label_probs)
    my_preds = pd.DataFrame(label_probs, columns=['ending0','ending1','ending2','ending3'])
    my_preds['pred'] = label_probs.argmax(1)
    my_preds.to_csv(args.output_file)


