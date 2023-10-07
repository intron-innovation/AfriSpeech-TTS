# AfriSpeech-TTS-100

#### Pan-African parameter-efficient multi-accent multi-speaker TTS


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


### How to Run the Code

1. Create a virtual environment `conda create -n afro_tts python=3.9`

2. Activate the virtual environment `conda activate afro_tts`

3. Install packages `pip install --upgrade transformers accelerate sentencepiece datasets[audio]`

5. Install requirements `pip3 install -r requirements.txt`

6. For Inference Run `python src/inference/inference.py`


### AfriSpeech Data Stats

- Total Number of Unique Speakers: 2,463
- Female/Male/Other Ratio: 57.11/42.41/0.48
- Data was first split on speakers. Speakers in Train/Dev/Test do not cross partitions

|  | Train | Dev | Test |
| ----------- | ----------- | ----------- | ----------- |
| # Speakers | 1466 | 247 | 750 |
| # Seconds | 624228.83 | 31447.09 | 67559.10 |
| # Hours | 173.4 | 8.74 | 18.77 |
| # Accents | 71 | 45 | 108 |
| Avg secs/speaker | 425.81 | 127.32 | 90.08 |
| Avg num clips/speaker | 39.56 | 13.08 | 8.46 |
| Avg num speakers/accent | 20.65 | 5.49 | 6.94 |
| Avg secs/accent | 8791.96 | 698.82 | 625.55 |
| # clips general domain | 21682 | 1407 | 2723 |
| # clips clinical domain | 36318 | 1824 | 3623 |


#### Country Stats

| Country | Clips | Speakers | Duration (seconds) | Duration (hrs) |
| ----------- | ----------- | ----------- | ----------- | ----------- |
| NG | 45875 | 1979 | 512646.88 | 142.40 |
| KE | 8304 | 137 | 75195.43 | 20.89 |
| ZA | 7870 | 223 | 81688.11 | 22.69 |
| GH | 2018 | 37 | 18581.13 | 5.16 |
| BW | 1391 | 38 | 14249.01 | 3.96 |
| UG | 1092 | 26 | 10420.42 | 2.89 |
| RW | 469 | 9 | 5300.99 | 1.47 |
| US | 219 | 5 | 1900.98 | 0.53 |
| TR | 66 | 1 | 664.01 | 0.18 |
| ZW | 63 | 3 | 635.11 | 0.18 |
| MW | 60 | 1 | 554.61 | 0.15 |
| TZ | 51 | 2 | 645.51 | 0.18 |
| LS | 7 | 1 | 78.40 | 0.02 |

#### Accent Stats

|  Accent | Clips | Speakers | Duration (s) | Country | Splits | 
| ----------- | ----------- | ----------- | ----------- | ----------- | ----------- |
| yoruba | 15407 | 683 | 161587.55 | US,NG | train,test,dev |
| igbo | 8677 | 374 | 93035.79 | US,NG,ZA | train,test,dev |
| swahili | 6320 | 119 | 55932.82 | KE,TZ,ZA,UG | train,test,dev |
| hausa | 5765 | 248 | 70878.67 | NG | train,test,dev |
| ijaw | 2499 | 105 | 33178.9 | NG | train,test,dev |
| afrikaans | 2048 | 33 | 20586.49 | ZA | train,test,dev |
| idoma | 1877 | 72 | 20463.6 | NG | train,test,dev |
| zulu | 1794 | 52 | 18216.97 | ZA,TR,LS | dev,train,test |
| setswana | 1588 | 39 | 16553.22 | BW,ZA | dev,test,train |
| twi | 1566 | 22 | 14340.12 | GH | test,train,dev |
| isizulu | 1048 | 48 | 10376.09 | ZA | test,train,dev |
| igala | 919 | 31 | 9854.72 | NG | train,test |
| izon | 838 | 47 | 9602.53 | NG | train,dev,test |
| kiswahili | 827 | 6 | 8988.26 | KE | train,test |
| ebira | 757 | 42 | 7752.94 | NG | train,test,dev |
| luganda | 722 | 22 | 6768.19 | UG,BW,KE | test,dev,train |
| urhobo | 646 | 32 | 6685.12 | NG | train,dev,test |
| nembe | 578 | 16 | 6644.72 | NG | train,test,dev |
| ibibio | 570 | 39 | 6489.29 | NG | train,test,dev |
| pidgin | 514 | 20 | 5871.57 | NG | test,train,dev |
| luhya | 508 | 4 | 4497.02 | KE | train,test |
| kinyarwanda | 469 | 9 | 5300.99 | RW | train,test,dev |
| xhosa | 392 | 12 | 4604.84 | ZA | train,dev,test |
| tswana | 387 | 18 | 4148.58 | ZA,BW | train,test,dev |
| esan | 380 | 13 | 4162.63 | NG | train,test,dev |
| alago | 363 | 8 | 3902.09 | NG | train,test |
| tshivenda | 353 | 5 | 3264.77 | ZA | test,train |
| fulani | 312 | 18 | 5084.32 | NG | test,train |
| isoko | 298 | 16 | 4236.88 | NG | train,test,dev |
| akan (fante) | 295 | 9 | 2848.54 | GH | train,dev,test |
| ikwere | 293 | 14 | 3480.43 | NG | test,train,dev |
| sepedi | 275 | 10 | 2751.68 | ZA | dev,test,train |
| efik | 269 | 11 | 2559.32 | NG | test,train,dev |
| edo | 237 | 12 | 1842.32 | NG | train,test,dev |
| luo | 234 | 4 | 2052.25 | UG,KE | test,train,dev |
| kikuyu | 229 | 4 | 1949.62 | KE | train,test,dev |
| bekwarra | 218 | 3 | 2000.46 | NG | train,test |
| isixhosa | 210 | 9 | 2100.28 | ZA | train,dev,test |
| hausa/fulani | 202 | 3 | 2213.53 | NG | test,train |
| epie | 202 | 6 | 2320.21 | NG | train,test |
| isindebele | 198 | 2 | 1759.49 | ZA | train,test |
| venda and xitsonga | 188 | 2 | 2603.75 | ZA | train,test |
| sotho | 182 | 4 | 2082.21 | ZA | dev,test,train |
| akan | 157 | 6 | 1392.47 | GH | test,train |
| nupe | 156 | 9 | 1608.24 | NG | dev,train,test |
| anaang | 153 | 8 | 1532.56 | NG | test,dev |
| english | 151 | 11 | 2445.98 | NG | dev,test |
| afemai | 142 | 2 | 1877.04 | NG | train,test |
| shona | 138 | 8 | 1419.98 | ZA,ZW | test,train,dev |
| eggon | 137 | 5 | 1833.77 | NG | test |
| luganda and kiswahili | 134 | 1 | 1356.93 | UG | train |
| ukwuani | 133 | 7 | 1269.02 | NG | test |
| sesotho | 132 | 10 | 1397.16 | ZA | train,dev,test |
| benin | 124 | 4 | 1457.48 | NG | train,test |
| kagoma | 123 | 1 | 1781.04 | NG | train |
| nasarawa eggon | 120 | 1 | 1039.99 | NG | train |
| tiv | 120 | 14 | 1084.52 | NG | train,test,dev |
| south african english | 119 | 2 | 1643.82 | ZA | train,test |
| borana | 112 | 1 | 1090.71 | KE | train |
| swahili ,luganda ,arabic | 109 | 1 | 929.46 | UG | train |
| ogoni | 109 | 4 | 1629.7 | NG | train,test |
| mada | 109 | 2 | 1786.26 | NG | test |
| bette | 106 | 4 | 930.16 | NG | train,test |
| berom | 105 | 4 | 1272.99 | NG | dev,test |
| bini | 104 | 4 | 1499.75 | NG | test |
| ngas | 102 | 3 | 1234.16 | NG | train,test |
| etsako | 101 | 4 | 1074.53 | NG | train,test |
| okrika | 100 | 3 | 1887.47 | NG | train,test |
| venda | 99 | 2 | 938.14 | ZA | train,test |
| siswati | 96 | 5 | 1367.45 | ZA | dev,train,test |
| damara | 92 | 1 | 674.43 | NG | train |
| yoruba, hausa | 89 | 5 | 928.98 | NG | test |
| southern sotho | 89 | 1 | 889.73 | ZA | train |
| kanuri | 86 | 7 | 1936.78 | NG | test,dev |
| itsekiri | 82 | 3 | 778.47 | NG | test,dev |
| ekpeye | 80 | 2 | 922.88 | NG | test |
| mwaghavul | 78 | 2 | 738.02 | NG | test |
| bajju | 72 | 2 | 758.16 | NG | test |
| luo, swahili | 71 | 1 | 616.57 | KE | train |
| dholuo | 70 | 1 | 669.07 | KE | train |
| ekene | 68 | 1 | 839.31 | NG | test |
| jaba | 65 | 2 | 540.66 | NG | test |
| ika | 65 | 4 | 576.56 | NG | test,dev |
| angas | 65 | 1 | 589.99 | NG | test |
| ateso | 63 | 1 | 624.28 | UG | train |
| brass | 62 | 2 | 900.04 | NG | test |
| ikulu | 61 | 1 | 313.2 | NG | test |
| eleme | 60 | 2 | 1207.92 | NG | test |
| chichewa | 60 | 1 | 554.61 | MW | train |
| oklo | 58 | 1 | 871.37 | NG | test |
| meru | 58 | 2 | 865.07 | KE | train,test |
| agatu | 55 | 1 | 369.11 | NG | test |
| okirika | 54 | 1 | 792.65 | NG | test |
| igarra | 54 | 1 | 562.12 | NG | test |
| ijaw(nembe) | 54 | 2 | 537.56 | NG | test |
| khana | 51 | 2 | 497.42 | NG | test |
| ogbia | 51 | 4 | 461.15 | NG | test,dev |
| gbagyi | 51 | 4 | 693.43 | NG | test |
| portuguese | 50 | 1 | 525.02 | ZA | train |
| delta | 49 | 2 | 425.76 | NG | test |
| bassa | 49 | 1 | 646.13 | NG | test |
| etche | 49 | 1 | 637.48 | NG | test |
| kubi | 46 | 1 | 495.21 | NG | test |
| jukun | 44 | 2 | 362.12 | NG | test |
| igbo and yoruba | 43 | 2 | 466.98 | NG | test |
| urobo | 43 | 3 | 573.14 | NG | test |
| kalabari | 42 | 5 | 305.49 | NG | test |
| ibani | 42 | 1 | 322.34 | NG | test |
| obolo | 37 | 1 | 204.79 | NG | test |
| idah | 34 | 1 | 533.5 | NG | test |
| bassa-nge/nupe | 31 | 3 | 267.42 | NG | test,dev |
| yala mbembe | 29 | 1 | 237.27 | NG | test |
| eket | 28 | 1 | 238.85 | NG | test |
| afo | 26 | 1 | 171.15 | NG | test |
| ebiobo | 25 | 1 | 226.27 | NG | test |
| nyandang | 25 | 1 | 230.41 | NG | test |
| ishan | 23 | 1 | 194.12 | NG | test |
| bagi | 20 | 1 | 284.54 | NG | test |
| estako | 20 | 1 | 480.78 | NG | test |
| gerawa | 13 | 1 | 342.15 | NG | test |