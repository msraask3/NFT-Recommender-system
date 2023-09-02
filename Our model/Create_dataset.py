import pandas as pd
import numpy as np
import scipy.sparse
import os
from collections import defaultdict
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pickle


def process_collection(COLLECTION):
    print(f"Processing collection: {COLLECTION}")

    df_collection = pd.read_csv(f"dataset/transactions/{COLLECTION}.csv")

    save_path = 'dataset/collections/'+COLLECTION+'/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # continue processing for the COLLECTION...

    print(f"Number of interactions in {COLLECTION} before filtering: {len(df_collection)}")

    # Exclude items that we do not have features data for.
    image = pd.read_csv(f'dataset/features_item/{COLLECTION}_image.csv', index_col=0)
    text = pd.read_csv(f'dataset/features_item/{COLLECTION}_text.csv', index_col=0)
    price = pd.read_csv(f'dataset/features_item/{COLLECTION}_prices.csv', index_col=0)
    transaction = pd.read_csv(f'dataset/features_item/{COLLECTION}_transactions.csv', index_col=0)
    indices = set(image.index).intersection(set(text.index)).intersection(set(price.index)).intersection(set(transaction.index))
    df_collection = df_collection[df_collection['Token ID'].isin(indices)]


    # Exclude users that we do not have features data for.
    df_feature = pd.read_csv(f'dataset/features_user/{COLLECTION}_user_features.csv', index_col=0)
    df_collection = df_collection[df_collection['Buyer'].isin(df_feature['Buyer'])]
    print(f"Number of interactions in {COLLECTION} after filtering: {len(df_collection)}")
    
    df_collection['Price'] = df_collection['Price'].astype(str)

    # convert 'Price' to the value before 'ETH'
    df_collection['Price'] = df_collection['Price'].str.extract(r'\((.*?)\)')
    df_collection['Price'] = df_collection['Price'].str.replace(',','').str.replace('$','').astype(float)
    # create a new variable 'Price_diff' which is the difference between the future price and the current price 
    df_collection['Price_diff'] = df_collection.groupby('Token ID')['Price'].diff(-1)
    df_collection['Price_diff'] = df_collection['Price_diff'].fillna(0)
    df_collection['Price_diff'] = df_collection['Price_diff'].apply(lambda x: -x)
    df_collection['Price_diff'] = df_collection['Price_diff'].apply(lambda x: 1 if x > 0 else 0)
    # Convert 'Buyer' column to string
    df_collection['Buyer'] = df_collection['Buyer'].astype(str)

    # create an np.array with 'Buyer'
    user = df_collection['Buyer'].values
    item = df_collection['Token ID'].values
    labels = df_collection['Price_diff'].values
    data = (user, item, labels)

    # save as npy file
    np.save(save_path + f'{COLLECTION}.npy', data)

    # print user length and item length
    print('user length: ', len(set(user)))
    print('item length: ', len(set(item)))
    print('inter length: ', len(labels))

    # save user length and item length as a dictionary
    dict = {'num_user': len(set(user)), 'num_item': len(set(item))}
    np.save(save_path + 'num_user_item.npy', dict)

    """
    *For RecBole*
    To use the same train, validation, and test sets when conducting baseline model experiments in RecBole, index information is stored.
    """

    # save df_collection as csv file
    recbole_path = '/home/seonmi/seonmi/nft/recsys/RecBole/dataset/transactions/'
    df_collection.reset_index(drop=True).to_csv(recbole_path + f'{COLLECTION}.csv', index=False)

    """
    Ensure that the indices for the user and item do not overlap with each other.
    We map indices using dict where the key is the original index and the value is the new index.

    We map the item indices to the range of [0, len(set(item))).
    We add len(set(item)) to the user indices.
    For example,
        Before:
            item: [5, 6, 8. 9, 10, 13, 15, 20, 21, 29]
            user: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        After:
            item: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            user: [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]

    """

    # 1) Map the item idx: start from 0.
    item_unique = np.unique(item)
    mapping_i = {}
    for i in range(len(item_unique)):
        mapping_i[item_unique[i]] = i

    # 2) Map the user idx: start from num_item.
    #   firstly, Change the user addresses to integers starting from 0 (e.g., 0x9137a5d195f0ab57e428c5a2be9bc8c4620445cb -> 0)
    #   then, add len(set(item)) to the user indices.
    user_unique = np.unique(user)
    mapping_u = {}
    for i in range(len(user_unique)):
        mapping_u[user_unique[i]] = i + len(set(item))

    # 3) Create inter
    user_ = np.array([mapping_u[u] for u in user])
    item_ = np.array([mapping_i[i] for i in item])
    inter = np.array([user_, item_, labels]).T
    inter = inter.astype(np.int64)
    print('num of interactions: ', inter.shape)

    df_feature = pd.read_csv(f'dataset/features_user/{COLLECTION}_user_features.csv', index_col=0)

    # Assume 'df_feature' is your DataFrame
    # df_feature['Buyer'] = df_feature.index  # Add 'Buyer' as a new column
    df_feature['Buyer'] = df_feature['Buyer'].apply(lambda x: mapping_u[x] if x in mapping_u else np.nan)
    df_feature = df_feature.dropna()

    # Convert 'Buyer' column to int
    df_feature['Buyer'] = df_feature['Buyer'].astype(int)

    print('num_user:', len(df_feature))

    # Set 'Buyer' as index
    df_feature = df_feature.set_index('Buyer')

    # Save df as npy file
    np.save(save_path + 'user_feat.npy', df_feature.to_numpy(), allow_pickle=True)

    """
    data split: use 40% of each user's interactions as validation and test data
    """
    # for each user, a random transaction and create a separate dataset with them
    valid_and_test = []
    random_idx_list = []
    for u in np.unique(inter[:,0]):
        num_sample = int(len(np.where(inter[:,0]==u)[0])*0.4) # 40% of the number of transactions
        random_idx = np.random.choice(np.where(inter[:,0]==u)[0], num_sample, replace=False)
        valid_and_test.extend(inter[random_idx])
        random_idx_list.extend(random_idx)
    valid_and_test = np.array(valid_and_test)

    """
    train
    """
    # create a separate dataset where inter not in random_idx_list
    train = np.delete(inter, random_idx_list, axis=0)
    # get list of indices inter-random_idx_list
    train_idx_list = list(set(range(len(inter))) - set(random_idx_list))

    """
    valid, test
    """
    # split valid_and_test into valid and test
    # split random_idx_list into 5:5
    valid_idx_list, test_idx_list = train_test_split(random_idx_list, test_size=0.5, random_state=42)
    valid = inter[valid_idx_list]
    test = inter[test_idx_list]

    # get ratio of train/inter, in percentage
    print(f'Train ratio: {len(train)/len(inter)*100:.2f}%')
    print(f'Valid and Test ratio: {len(valid_and_test)/len(inter)*100:.2f}%')

    """
    *For RecBole*
    To use the same train, validation, and test sets when conducting baseline model experiments in RecBole, index information is stored.
    """

    # create a list of lists, where each list contains indices of train, validation, and test sets
    indices = [train_idx_list, valid_idx_list, test_idx_list]

    # save indices as pkl file
    recbole_path = f'RecBole/dataset/collections/{COLLECTION}/'
    with open(recbole_path + 'split_indices.pkl', 'wb') as f:
        pickle.dump(indices, f)


        """
    preprocessing valid data
    """
    # using valid, create a dict where keys are unique users and values are items
    valid_dict = {}
    for i in range(len(valid)):
        if valid[i][0] in valid_dict:
            valid_dict[valid[i][0]].append(valid[i][1])
        else:
            valid_dict[valid[i][0]] = [valid[i][1]]

    # show the first five items in valid_dict
    list(valid_dict.items())[:5]

    """
    Extract the item index in the order of the most traded (popular).
    """

    # concat all values in valid_dict as a list
    valid_list = []
    for i in valid_dict.values():
        valid_list += i

    # value count valid_list and sort values
    value_counts = pd.Series(valid_list).value_counts().sort_values(ascending=False)

    # extract indices of value_counts
    indices = value_counts.index

    # save indices as npy
    np.save(save_path+'indices_valid.npy', indices, allow_pickle=True)


    """
    Convert to the form required by the model
    e.g., 12656: [7314, 4820, 6304] -> list([12656, 7314, 4820, 6304])
    """

    # Create an empty numpy array with dtype 'object'
    valid_array = np.empty(len(valid_dict), dtype=object)

    # Assign the lists directly to the elements of the array
    for i, (key, val) in enumerate(valid_dict.items()):
        # include key in the list
        valid_array[i] = [key] + val

    # show the first five items in valid_array
    valid_array[:5]


    """
    preprocessing test data
    """

    # using test, create a dict where keys are unique users and values are items
    test_dict = {}
    for i in range(len(test)):
        if test[i][0] in test_dict:
            test_dict[test[i][0]].append(test[i][1])
        else:
            test_dict[test[i][0]] = [test[i][1]]

    # show the first five items in test_dict
    list(test_dict.items())[:5]


    """
    Extract the item index in the order of the most traded (popular).
    """

    # concat all values in test_dict as a list
    test_list = []
    for i in test_dict.values():
        test_list += i

    # value count test_list and sort values
    value_counts = pd.Series(test_list).value_counts().sort_values(ascending=False)

    # extract indices of value_counts
    indices = value_counts.index

    # save indices as npy
    np.save(save_path+'indices_test.npy', indices, allow_pickle=True)

    """
    Convert to the form required by the model
    e.g., 12656: [7314, 4820, 6304] -> list([12656, 7314, 4820, 6304])
    """

    # Create an empty numpy array with dtype 'object'
    test_array = np.empty(len(test_dict), dtype=object)

    # Assign the lists directly to the elements of the array
    for i, (key, val) in enumerate(test_dict.items()):
        # include key in the list
        test_array[i] = [key] + val

    # show the first five items in test_array
    test_array[:5]


    # save train, valid, test as npy file
    np.save(save_path+'train.npy', train, allow_pickle=True)
    np.save(save_path+'val.npy', valid_array, allow_pickle=True)
    np.save(save_path+'test.npy', test_array, allow_pickle=True)

    # first column of inter is user
    # second column of inter is item

    # create a dict where keys are user and values are items
    adj_dict = {}
    for i in range(len(inter)):
        if inter[i][0] in adj_dict:
            adj_dict[inter[i][0]].append(inter[i][1])
        else:
            adj_dict[inter[i][0]] = [inter[i][1]]

    # show the first five items in adj_dict
    print(list(adj_dict.items())[:5])

    # save adj_dict as npy file
    np.save(save_path+'adj_dict.npy', adj_dict, allow_pickle=True)

    # print image, text, price shape
    print('Before')
    print('image shape: ', image.shape)
    print('text shape: ', text.shape)
    print('price shape: ', price.shape)
    print('transaction shape: ', transaction.shape)
    print('')

    """
    Keep only the items that appear in the inter
    """
    # for dataset image, text, price, filter rows whose indices are in item_unique
    item_unique = np.unique(item)
    image = image.loc[image.index.isin(item_unique)]
    text = text.loc[text.index.isin(item_unique)]
    price = price.loc[price.index.isin(item_unique)]
    transaction = transaction.loc[transaction.index.isin(item_unique)]

    """
    Change the item index to start from 0
    """
    # convert indices using mapping_i
    image.index = image.index.map(mapping_i)
    text.index = text.index.map(mapping_i)
    price.index = price.index.map(mapping_i)
    transaction.index = transaction.index.map(mapping_i)

    # print image, text, price shape
    print('After')
    print('image shape: ', image.shape)
    print('text shape: ', text.shape)
    print('price shape: ', price.shape)
    print('transaction shape: ', transaction.shape)

    # assert that the indices of image, text, price are the same, regardless of the order
    assert np.array_equal(np.sort(image.index.values), np.sort(text.index.values))
    assert np.array_equal(np.sort(image.index.values), np.sort(price.index.values))
    assert np.array_equal(np.sort(image.index.values), np.sort(transaction.index.values))

    # save df as npy file
    np.save(save_path+'image_feat.npy', image)
    np.save(save_path+'text_feat.npy', text)
    np.save(save_path+'price_feat.npy', price)
    np.save(save_path+'transaction_feat.npy', transaction)
            # return when done, if anything needs to be returned.


# collections to process
collections = ['bayc', 'coolcats', 'doodles', 'meebits']

# process all collections
for collection in collections:
    process_collection(collection)
