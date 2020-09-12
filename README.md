# PixGen
Our team's web app aids physicians and other healthcare providers in uploading pictures of their patients’ skin lesions, creating an open-source database of skin conditions in patients with darker skin tones. We then plan to use this database in downstream machine learning algorithms, including training a DCGAN to generate even more examples of skin conditions in darker skin tones. Our goal is to increase the visibility of skin conditions in all skin tones and remove cognitive biases that contribute to poorer health outcomes in minorities.

## Inspiration
In the news, there has been a lot of talk about the dermatological symptoms associated with COVID-19 infection, even if the individual is otherwise asymptomatic. These symptoms may include rash, blisters, or itchy hives. While researching this topic, our team began to realize that **skin conditions related to COVID-19 may look different between patients with fair skin and those with darker skin**. A paper by Lester et al demonstrated that in a systematic review of pictures in scientific articles describing skin manifestations associated with COVID-19, 93% (120 out of 130) were taken with patients with the three fairest skin tones (Types I-III). 6% (7 out of 130) showed patients with Type IV skin, and **there was no representation of the darkest skin tones (Type V and VI)**. This can lead to cognitive biases that contribute to underdiagnosis of COVID-19 infection in patients with darker skin that are otherwise asymptomatic.

This issue isn't just limited to COVID-19 though. Turns out, **many other skin conditions that present differently in patients with dark skin compared to those with fair skin**.

The following is an example of atopic dermatitis in an infant with dark skin compared to one with fair skin.
We have decided to make a web application where clinicians can decide to add and request images of skins lesions from POC.
![](https://res.cloudinary.com/devpost/image/fetch/s--6LtOQDI8--/c_limit,f_auto,fl_lossy,q_auto:eco,w_900/https://www.statnews.com/wp-content/uploads/2020/07/Screen-Shot-2020-07-03-at-7.26.08-AM-e1595268565603.png)
**Figure 1.** Atopic dermatitis in an infant with dark skin (left) compared to one with fair skin (right)

## Demo
[Figma of Web App Vision](https://www.figma.com/proto/Ujp8zPKYjvmOqxfkyTN3Wx/Invictus?node-id=1%3A130&scaling=min-zoom)

## Built with
- Tensorflow (TF) / TF Keras API
- Firebase 
- HTML
- CSS
- JS


## Installation

```
git clone https://github.com/gitUmaru/PixGen.git
cd ./Pixgen

pip install virtualenv

virtualenv env

env\scripts\activate

pip install tensorflow==2.*
pip install matplotlib
pip install tensorflow_hub
```
Change code inside the `style_transfer.py` and `dcgan.py` files for the correct image path directory.

Then run `python ./style_transfer.py` and `python ./dcgan.py`

## Results

![](https://github.com/gitUmaru/PixGen/blob/master/dcgan-pipeline/lesion.jpg?raw=true)|![](https://github.com/gitUmaru/PixGen/blob/master/doc_imgs/stylized-image.png?raw=true)|
:-------------------------:|:-------------------------:|
**Figure 2.1.** Light Skin Lesion from HAM10000| **Figure 2.2.** Dark Skin Lesion from Neural Style Transfer|


![](https://github.com/gitUmaru/PixGen/blob/master/doc_imgs/dcgan4x4.gif?raw=true)

**Figure 3.** DCGAN Image Generation GIF of skin lesions (Black and white)

We won 3<sup>rd</sup> place at John Hopkin's School od Medicine MedHacks 2020

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[MIT](https://choosealicense.com/licenses/mit/)
