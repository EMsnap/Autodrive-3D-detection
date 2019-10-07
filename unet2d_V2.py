from datetime import datetime
from lyft_dataset_sdk.lyftdataset import LyftDataset
from utils.bev_generator import *
from model.dataset import BEVImageDataset
from model.unet import *
import pandas as pd
import glob

# Disable multiprocesing for numpy/opencv. We already multiprocess ourselves, this would mean every subprocess produces
# even more threads which would lead to a lot of context switching, slowing things down a lot.

os.environ["OMP_NUM_THREADS"] = "1"


# os.symlink("/media/pzr/软件/kaggle/3D/train_images", 'images')
# os.symlink("/media/pzr/软件/kaggle/3D/train_maps", 'maps')
# os.symlink("/media/pzr/软件/kaggle/3D/train_lidar", 'lidar')

# Our code will generate data, visualization and model checkpoints, they will be persisted to disk in this folder
DATA_PATH = '/media/pzr/软件/kaggle/3D/train_data'
ARTIFACTS_FOLDER = "/media/pzr/软件/kaggle/3D/artifacts/"

level5data = LyftDataset(data_path='.', json_path=DATA_PATH, verbose=True)
os.makedirs(ARTIFACTS_FOLDER, exist_ok=True)


classes = ["car", "motorcycle", "bus", "bicycle", "truck", "pedestrian", "other_vehicle", "animal", "emergency_vehicle"]

records = [(level5data.get('sample', record['first_sample_token'])['timestamp'],
            record) for record in level5data.scene]

entries = []

for start_time, record in sorted(records):
    # 通过 first_sample_token 取得sample表
    # sample表示一个Scene中的一个snapshot(相当于一个瞬间的记录，每个scene有126个sample，所以一共有126 * 180 = 22680 个sample
    start_time = level5data.get('sample', record['first_sample_token'])['timestamp'] / 1000000 # start_time: float
    token = record['token']
    name = record['name']
    date = datetime.utcfromtimestamp(start_time)
    host = "-".join(record['name'].split("-")[:2])
    first_sample_token = record["first_sample_token"]

    entries.append((host, name, date, token, first_sample_token))

df = pd.DataFrame(entries, columns=["host", "scene_name", "date",
                                    "scene_token", "first_sample_token"])

# host_count_df = df.groupby("host")['scene_token'].count()
host_count_df = df.groupby("host")['scene_token'].count()
print(df.groupby("host")['scene_token'])
print(host_count_df)

validation_hosts = ["host-a007", "host-a008", "host-a009"]

validation_df = df[df["host"].isin(validation_hosts)]
vi = validation_df.index
train_df = df[~df.index.isin(vi)]

print(len(train_df), len(validation_df), "train/validation split scene counts")

print(train_df.first_sample_token.values[0])

# Creating input and targets
# Let's load the first sample in the train set. We can use that to test the functions
# we'll define next that transform the data to the format we want to input into the model we are training.

# Let's try it with some example values

voxel_size = (0.4, 0.4, 1.5)
z_offset = -2.0
bev_shape = (336, 336, 3)


def visualize_lidar_of_sample(sample_token, axes_limit=80):
    sample = level5data.get("sample", sample_token)
    sample_lidar_token = sample["data"]["LIDAR_TOP"]
    level5data.render_sample_data(sample_lidar_token, axes_limit=axes_limit)


# We scale down each box so they are more separated when projected into our coarse voxel space.
box_scale = 0.8

# "bev" stands for birds eye view
train_data_folder = os.path.join(ARTIFACTS_FOLDER, "bev_train_data")
validation_data_folder = os.path.join(ARTIFACTS_FOLDER, "./bev_validation_data")

NUM_WORKERS = os.cpu_count()

# Run it once for generating training data, save it for further use

# generate_training_data(level5data, train_df=train_df, train_data_folder=train_data_folder, validation_df=validation_df,
#                         validation_data_folder=validation_data_folder, NUM_WORKERS=NUM_WORKERS, box_scale=box_scale,
#                         voxel_size=voxel_size, z_offset=z_offset, bev_shape=bev_shape, classes=classes)


input_filepaths = sorted(glob.glob(os.path.join(train_data_folder, "*_input.png")))
target_filepaths = sorted(glob.glob(os.path.join(train_data_folder, "*_target.png")))

train_dataset = BEVImageDataset(input_filepaths, target_filepaths)

im, target, sample_token = train_dataset[1]
im = im.numpy()
target = target.numpy()

plt.figure(figsize=(16, 8))

target_as_rgb = np.repeat(target[..., None], 3, 2)
# Transpose the input volume CXY to XYC order, which is what matplotlib requires.
plt.imshow(np.hstack((im.transpose(1, 2, 0)[..., :3], target_as_rgb)))
plt.title(sample_token)
plt.show()

visualize_lidar_of_sample(sample_token)

# We weigh the loss for the 0 class lower to account for (some of) the big class imbalance.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class_weights = torch.from_numpy(np.array([0.2] + [1.0]*len(classes), dtype=np.float32))
class_weights = class_weights.to(device)

batch_size = 3
epochs = 30

# Note: We may be able to train for longer and expect better results,
# the reason this number is low is to keep the runtime short.

model = get_unet_model(num_output_classes=len(classes) + 1)
model = model.to(device)

optim = torch.optim.Adam(model.parameters(), lr=1e-3)
dataloader = torch.utils.data.DataLoader(train_dataset, batch_size, shuffle=True,
                                         num_workers=os.cpu_count() * 2)

all_losses = []

for epoch in range(1, epochs + 1):
    print("Epoch", epoch)

    epoch_losses = []
    progress_bar = tqdm(dataloader)

    for ii, (X, target, sample_ids) in enumerate(progress_bar):
        X = X.to(device)  # [N, 3, H, W]
        target = target.to(device)  # [N, H, W] with class indices (0, 1)
        prediction = model(X)  # [N, 2, H, W]
        loss = F.cross_entropy(prediction, target, weight=class_weights)

        optim.zero_grad()
        loss.backward()
        optim.step()

        epoch_losses.append(loss.detach().cpu().numpy())

        #if ii == 0:
        #    visualize_predictions(X, prediction, target)

    print("Loss:", np.mean(epoch_losses))
    all_losses.extend(epoch_losses)

    checkpoint_filename = "unet_checkpoint_epoch_{}.pth".format(epoch)
    checkpoint_filepath = os.path.join(ARTIFACTS_FOLDER, checkpoint_filename)
    torch.save(model.state_dict(), checkpoint_filepath)

plt.figure(figsize=(12, 12))
plt.plot(all_losses, alpha=0.75)
plt.show()





































