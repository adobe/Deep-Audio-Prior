# Deep Audio Prior - Pytorch Implementation

Our deep audio prior can enable several audio applications: blind sound source separation, interactive mask-based editing, audio textual synthesis, and audio watermarker removal.

### Blind source separation

Our DAP-based BSS model can separate individual sound sources from a sound mixture without using any external training data.
For evaluation, we compose a 2-channel input sound with two individual sounds: s1 and s2, then we generate a sound mixture: s_mix = s1+s2.
 ```bash
    $ cd ~/code/
    $ python dap_sep.py --input_mix data/sep/violin_basketball.wav --output output/sep
 ```
The separated sounds and other intermediate results can be found in the "code/output/sep" folder.

### Interactive mask-based editing

User can interact with generated masks for audio sources to further improve separation results. 

 ```bash
    $ cd ~/code/
    $ python dap_mask_1st.py --input_mix xxx --out data/mask/ckpt
    $ prepare a binary map to deactivate regions in a generated mask and save it into "data/mask/ckpt"
    $ python dap_mask_2rd.py --input_mix xxx --dea_map xxx --dea_map_id xxx --output xxxx
 ```
For the second round with mask interaction, we have two additional parameters: dea_map and dea_map_id, which refer to an annotated binary map and the corresponding audio source ID.
We provide one example that refines separation results from a dog and violin mixture with an annotated deactivation binary map for the dog sound:
```bash
    $ cd ~/code/
    $ python dap_mask_2rd.py --input_mix data/mask/violin_dog.wav --dea_map data/mask/ckpt/mask2_dea.npy --dea_map-id 2 --output output/mask
 ```

### Audio Textual Synthesis

DAP can be used to synthesize audio textures.
 ```bash
    $ cd ~/code/
    $ python dap_audio_synthesis.py --input data/synthesis/water.wav --output output/sysnthesis
 ```
 

### Co-separation/audio watermarker removal

DAP can also be successfully applied to address audio watermarker removal with co-separation. Given 3 sounds with audio watermarkers, our cosep model can generate 3 individual music sounds and the corresponding watermarker.
 ```bash
    $ cd ~/code/
    $ python dap_cosep.py --input1 data/cosep/audiojungle/01.mp3 --input2 data/cosep/audiojungle/02.mp3 --input3 data/cosep/audiojungle/03.mp3 --output output/cosep
 ```
 

### Citation

<pre><code>@InProceedings{dap2019,
  author={Yapeng Tian, Chenliang Xu, and Dingzeyu Li},
  title={Deep Audio Prior},
  year = {2019}
}
