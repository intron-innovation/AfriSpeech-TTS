# AfriSpeech-TTS

#### African Digital Voices: Pan-African parameter-efficient multi-accent multi-speaker TTS

[By Intron Innovation](https://www.intron.io)

[By BioRAMP](https://www.bioramp.org)

Contributor List: []

#### Progress
- [x] Preprocess data
- [x] Setup repo and starter scripts
- [] Fine-tune backbone
- [] Create accent-aware architecture


#### Abstract [draft]

Recent advances in speech synthesis have enabled may useful useful applications like audio directions in Google Maps
, voice cloning, screen readers, and automated content  generation on social media platforms like Tik-tok and
 Instagram. However many of these systems are dominated by voices sourced from data rich geographies with personas
  representative of their source data. Although 3000 of the world's languages are domiciled in Africa, African voices
   and personas are extremely under-represented in these systems. Recent transformer-based state-of-the-art models
    are large, compute-intensive, and data hungry. Training a representative diversity of models for multiple African
     accents in resource constrained settings is infeasible and prohibitive. Parameter-Efficient approaches therefore
      gained popularity using only 1.2% to 0.8% of original trainable parameters to achieve competitive performance
       in voice synthesis. We present Afro-TTS, the first pan-African accented English speech synthesis system able
        to generate speech in 100+ African accents, with 200 personas representing the rich phonological diversity
         across the continent for downstream localized application in Education, Public Health, and Automated Content
          Creation.



### How to Access the Data

Train, dev, and test sets have been uploaded to an s3 bucket for public access.
Here are the steps to access the data

1. If not installed already, download and install `awscli` for your 
platform (linux/mac/windows) following the instructions [here](https
://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html) 

2. Create a download folder e.g. `mkdir AfriSpeech-TTS`

3. Request aws credentials to access the data by sending an email
with title "AfriSpeech-TTS S3 Credentials Request" to tobi@intron.io

4. Once you receive credentials, cd into your data directory `cd AfriSpeech-TTS`

5. Type `aws configure` at the command line, hit enter, and fill in the following based on the credentials you receive.
    ```
    AWS Access Key ID [None]: <ACCESS-KEY-ID-FROM-CREDENTIALS-SENT>
    AWS Secret Access Key [None]: <SECRET-KEY-FROM-CREDENTIALS-SENT>
    Default region name [None]: eu-west-2
    Default output format [None]: <leave this blank, hit enter>
    ```

6. Run `aws s3 cp s3://intron-open-source/AfriSpeech-TTS . --recursive` to download all the audio

7. Download may take over 2hrs depending on your bandwidth. Train set: 28,565, 103G; Dev set: 3,330, 5.2G; Test set: 4,161, xG



### How to Run the Code

#### For Inference

1. Create a virtual environment `conda create -n afro_tts python=3.9`

2. Activate the virtual environment `conda activate afro_tts`

3. Install packages `pip install --upgrade transformers accelerate sentencepiece datasets[audio]`

5. Install requirements `pip3 install -r requirements.txt`

6. For Inference Run `python src/inference/inference.py --model_id_or_path microsoft/speecht5_tts`

## Training 
```bash 
python3 src/train/train.py -c src/config/config_speechT5.ini
```

### AfriSpeech Data Stats

- Total Number of Unique Speakers: 751
- Female/Male/Other Ratio: 54.45/44.36/1.19
- Data was first split on speakers. Speakers in Train/Dev/Test do not cross partitions

|  | Train | Dev | Test |
| ----------- | ----------- | ----------- | ----------- |
| # Speakers | 600 | 76 | 75 |
| # Seconds | 383411.40 | 47506.20 | 57492.25 |
| # Hours | 106.5 | 13.2 | 15.97 |
| # Accents | 72 | 25 | 25 |
| Avg secs/speaker | 639.02 | 625.08 | 766.56 |
| Avg num clips/speaker | 47.61 | 43.82 | 55.48 |
| Avg num speakers/accent | 8.33  | 3.04 | 3.0 |
| Avg secs/accent | 5325.16 | 1900.25 | 2299.69 |


#### Country Stats

| Country | Clips | Speakers | Accents  |  Duration (seconds) | Duration (hrs) |
| ----------- | ----------- | ----------- | ----------- | ----------- | ----------- |
| NG | 25564 | 549 | 48 | 359469.91 | 99.85 |
| KE | 5307 | 58 | 8 | 59231.35 | 16.45 |
| ZA | 4279 | 125 | 20 | 59066.38 | 16.41 |
| GH | 727 | 4 | 3 | 8200.78 | 2.28 |
| ZW | 47 | 6 | 3 | 718.90 | 0.20 |
| RW | 40 | 1 | 1 | 524.41 | 0.15 |
| SL | 38 | 1 | 1 | 504.21 | 0.14 |
| UG | 26 | 2 | 1 | 302.59 | 0.08 |
| ZM | 8 | 1 | 1 | 149.86 | 0.04 |
| US | 1 | 1 | 1 | 8.18 | 0.00 |

#### Accent Stats

|  Accent | Clips | Speakers | Duration (s) | Country | Splits | 
| ----------- | ----------- | ----------- | ----------- | ----------- | ----------- |
| Swahili | 2993 | 40 | 36794.19 | train,test,dev |
| Zulu | 1735 | 38 | 23158.02 | train,test,dev |
| Xhosa | 116 | 9 | 1427.93 | train,test,dev |
| Afrikaans | 503 | 9 | 7482.53 | train,test |
| Swahili, Kamba | 498 | 2 | 4101.61 | train |
| Shona | 276 | 10 | 3111.83 | train,dev |
| Tswana | 538 | 12 | 7083.35 | train,test,dev |
| Hindi | 19 | 1 | 326.53 | train |
| Southern Sotho | 86 | 9 | 1120.94 | train,test,dev |
| Siswati | 47 | 6 | 1008.68 | train,dev |
| Pedi | 138 | 6 | 1844.98 | train,dev |
| Akan | 586 | 2 | 6668.93 | train |
| Tsonga | 12 | 2 | 124.15 | train,test |
| Luganda | 26 | 2 | 302.59 | train |
| Ndebele | 29 | 3 | 419.51 | train |
| Luhya | 52 | 1 | 705.17 | train |
| Swati | 6 | 1 | 77.64 | train |
| Portuguese | 2 | 1 | 30.24 | train |
| Twi | 98 | 1 | 1070.99 | train |
| Kikuyu | 84 | 2 | 1267.97 | train |
| South African English | 78 | 1 | 1185.14 | train |
| Ga | 43 | 1 | 460.86 | train |
| Northern Sotho | 23 | 2 | 405.53 | train,dev |
| Kikuyu, Swahili | 2 | 1 | 14.2 | train |
| Hausa | 8208 | 154 | 110710.67 | train,test,dev |
| Meru | 70 | 1 | 912.74 | train |
| Venda, Tsonga | 63 | 1 | 1119.13 | train |
| Venda | 17 | 1 | 262.38 | train |
| Yoruba | 8181 | 170 | 114025.69 | train,test,dev |
| Kinyarwanda | 40 | 1 | 524.41 | train |
| Hausa, Yoruba, Pidgin | 268 | 2 | 4427.57 | train,test |
| Ijaw | 1096 | 22 | 17340.83 | train,test,dev |
| Ibibio | 518 | 13 | 6742.1 | train,test,dev |
| Igbo | 2209 | 50 | 32033.57 | train,test,dev |
| Igala | 587 | 10 | 11252.44 | train,dev |
| Benin | 189 | 2 | 3182.55 | train |
| Obolo | 2 | 1 | 12.57 | train |
| Idoma | 728 | 14 | 10673.23 | train,test,dev |
| Urhobo | 368 | 8 | 5555.68 | train |
| Nupe | 57 | 3 | 973.86 | train,dev |
| Gwari | 30 | 1 | 305.3 | train |
| Tula | 17 | 2 | 167.67 | train,test |
| Epie | 28 | 2 | 361.54 | train |
| Tiv | 189 | 6 | 2572.66 | train |
| Fulani | 102 | 4 | 2095.19 | train,test,dev |
| Isoko | 176 | 3 | 2557.76 | train,dev |
| Igede | 17 | 1 | 304.66 | train |
| Bette | 22 | 2 | 284.87 | train,test |
| Lunguda | 2 | 1 | 42.36 | train |
| Etsako | 37 | 3 | 400.2 | train,dev |
| Ogbia | 6 | 1 | 78.48 | train |
| Utonkon | 141 | 2 | 2559.5 | train |
| Bajju | 17 | 1 | 182.39 | train |
| Kanuri | 36 | 3 | 397.78 | train,test |
| Hausa, Higgi, Pidgin | 132 | 1 | 2221.02 | train |
| Dera | 15 | 1 | 166.64 | train |
| English | 149 | 1 | 2137.42 | train |
| Gerawa | 21 | 1 | 272.22 | train |
| Anaang | 3 | 1 | 31.14 | train |
| Yoruba, Pidgin | 120 | 1 | 2070.34 | train |
| Hausa, Fulani | 4 | 1 | 33.25 | train |
| Ikulu | 6 | 1 | 54.91 | train |
| Pidgin | 60 | 7 | 809.06 | train,test,dev |
| Kagoma | 1 | 1 | 21.5 | train |
| Etuno | 146 | 1 | 2025.85 | train |
| Ebira | 168 | 4 | 2285.19 | train |
| Ogoni | 1 | 1 | 7.56 | train |
| Bekwarra | 32 | 1 | 416.87 | train |
| Ogba | 2 | 1 | 34.22 | train |
| Esan | 80 | 2 | 1023.47 | train |
| Ika | 1 | 1 | 12.24 | train |
| Chichewa | 8 | 1 | 149.86 | test |
| Kamba, Swahili | 400 | 1 | 3827.56 | test |
| Tshivenda | 13 | 1 | 217.05 | test |
| Jarawa | 25 | 1 | 319.4 | test |
| Kubi | 1 | 1 | 10.19 | test |
| Otjiherero | 5 | 1 | 61.7 | dev |
| Nyandang | 68 | 1 | 774.0 | dev |
| Hausa, Pidgin | 14 | 1 | 223.89 | dev |
| Unknown | 3170 | 69 | 36950.02 | train,test,dev |
