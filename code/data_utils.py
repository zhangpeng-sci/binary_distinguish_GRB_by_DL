import numpy as np
from keras.utils import to_categorical


def normalize_item(item):
    _range = np.max(item) - np.min(item)
    return (item - np.min(item)) / _range


def normalize_data_array(data_array):
    result = []
    for data_item in data_array:
        result.append(normalize_item(data_item))
    return np.array(result)


def get_train_val_data(dataset_dir, time_bin):
    def get_y_data(info_array):
        class_name_array = info_array[:, 2]
        temp_y_pre = np.ones(class_name_array.size)
        temp_y_pre[class_name_array == "normal"] = 0
        return temp_y_pre

    if time_bin not in ["64ms", "128ms", "256ms"]:
        raise Exception("The time_bin should in [64ms,128ms,256ms]")

    train_x = np.load(f"{dataset_dir}train_count_map_{time_bin}.npy")
    train_info = np.load(f"{dataset_dir}train_info.npy")
    train_y = get_y_data(train_info)

    val_x = np.load(f"{dataset_dir}validate_count_map_{time_bin}.npy")
    val_info = np.load(f"{dataset_dir}validate_info.npy")
    val_y = get_y_data(val_info)

    train_x = train_x[:, :, :, np.newaxis]
    val_x = val_x[:, :, :, np.newaxis]

    train_x = normalize_data_array(train_x)
    val_x = normalize_data_array(val_x)

    class_num = 2
    train_y = to_categorical(train_y, class_num)
    val_y = to_categorical(val_y, class_num)

    return (train_x, train_y, train_info), (val_x, val_y, val_info)


def get_test_data(dataset_dir, time_bin):
    def get_y_data(info_array):
        class_name_array = info_array[:, 2]
        temp_y_pre = np.ones(class_name_array.size)
        temp_y_pre[class_name_array == "normal"] = 0
        return temp_y_pre

    if time_bin not in ["64ms", "128ms", "256ms"]:
        raise Exception("The time_bin should in [64ms,128ms,256ms]")

    test_x = np.load(f"{dataset_dir}test_count_map_{time_bin}.npy")
    test_info = np.load(f"{dataset_dir}test_info.npy")
    test_y = get_y_data(test_info)

    test_x = test_x[:, :, :, np.newaxis]
    test_x = normalize_data_array(test_x)

    class_num = 2
    test_y = to_categorical(test_y, class_num)

    return (test_x, test_y, test_info)
