!pip install tensorflow_datasets
import tensorflow_datasets as tfds

def download_imdb_data():
    % % capture
    train_data, test_data = tfds.load(name="imdb_reviews", split=["train", "test"],
                                      batch_size=-1, as_supervised=True)

    train, _ = tfds.as_numpy(train_data)
    test, _ = tfds.as_numpy(test_data)

def bytes_to_str(array):
    '''
    Convert bytes to string since IMDB download above downloads as bytes.
    '''
    copy = array.copy()
    for i in range(len(array)):
        copy[i] = array[i].decode('utf-8')
    return copy

def fix_nltk_download_ssl_error():
    import ssl

    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context
    # try running e.g.
    #nltk.download('punkt')