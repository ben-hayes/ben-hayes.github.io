---
layout: page
title: neural waveshaping synthesis
description: real-time neural audio synthesis in the waveform domain
img: /assets/img/newt_shapers.png
importance: 1
category: work
---

<style>
.example {
    color: #999999
}
</style>

<div style="margin:auto; text-align:center;">
    <h4>
        <a href="{{ '/assets/pdf/nws_arxiv.pdf' | relative_url }}" target="_blank">paper</a> •
        <a href="https://github.com/ben-hayes/neural-waveshaping-synthesis/" target="_blank">code</a> • 
        <a href="https://colab.research.google.com/github/ben-hayes/neural-waveshaping-synthesis/blob/main/colab/NEWT_Timbre_Transfer.ipynb" target="_blank">colab</a> • 
        <a href="#audio-examples">audio</a>
    </h4>
    <p>
    by <em>Ben Hayes, Charalampos Saitis, György Fazekas</em>
    </p>
</div>

We're excited to present our work on ***Neural Waveshaping Synthesis***, a method for efficiently synthesising audio waveforms with neural networks.
Our approach adds to the growing family of differentiable digital signal processing (DDSP) [^1] methods by learning waveshaping functions with neural networks.

This page is a supplement to our paper, accepted for publication at [ISMIR 2021](https://ismir2021.ismir.net/).

### what is waveshaping synthesis?

Musical audio often has a harmonic structure.
In other words, it consists of multiple components oscillating at integer multiples of a single fundamental frequency, called harmonics.
Many audio synthesis methods focus on producing such harmonics, and *waveshaping synthesis* [^2] is one of these approaches.

In brief, we take a simple sinusoid, $$\cos\omega n$$, and we pass it through a nonlinear function, which produces a sum of harmonics of that sinusoid:

$$f(\cos \omega n) = \sum_{k=1}^{\infty} h_k \cos \omega n.$$

This is the principle of *harmonic distortion*, and by designing an appropriate function $$f$$ we can produce any combination of harmonics we require. 
<span class="example">Note: this is a big simplification! We are assuming ideal conditions — in practice, aliasing means the resulting signal is often more complex.</span>

If we apply $$f$$ to a more complex signal, $$y[n]$$, we encouter a phenomenon called *intermodulation distortion*, which produces frequencies at $$a\omega_1 \pm b\omega_2, \; \forall a,b \in \mathbb{Z}^+$$, for input frequencies $$\omega_1$$ and $$\omega_2$$.
This means when $$y[n]$$ is purely harmonic, $$f(y[n])$$ will be too.

With this knowledge, we can produce a steady signal consisting of multiple harmonics.
If we want to vary the timbre over time, we can change the amplitude of the signal we pass into $$f$$.
We do this using the distortion index $$a[n]$$.
We also introduce the normalising coefficient $$N[n]$$ which allows us to vary the timbre separately from the amplitude of the final signal.
Putting all this together, we can define a simple waveshaping synthesiser:

$$x[n] = N[n]f(a[n] \cos \omega n)$$

As you might have guessed, the tricky part here is choosing the function $$f$$.
In Marc Le Brun's original formulation[^2], the function is defined as a sum of Chebyshev polynomials of the first kind, giving a precise combination of harmonics when $$a[n]=1$$.

But what happens when we vary $$a[n]$$?
As we know, the balance of harmonics also varies, but with the Chebyshev polynomial design method we have no control over this variation.
This means if we want to model the temporal evolution of a specific target timbre, we're out of luck!

Or are we?


### NEWT: the **ne**ural **w**aveshaping uni**t**

The NEWT is a neural network structure that performs waveshaping on an exciter signal.
Instead of manually designing our shaping function $$f$$, however, the NEWT learns it from unlabelled audio.
In particular, the NEWT learns an implicit neural representation[^3] of the shaping function $$f$$ using a sinusoidal multi-layer perceptron (MLP).
And it turns out that we can get good results using *tiny* MLPs to represent our shaping functions -- only 169 parameters each!

<div class="row">
    <div style="width: 20%">
    </div>
    <div class="col-sm mt-3 mt-md-0">
        <img class="img-fluid rounded " src="{{ '/assets/img/newt_new.png' | relative_url }}" alt="" title="The architecture of the Neural Waveshaping Unit"/>
    </div>
    <div style="width: 20%">
    </div>
</div>
<div class="caption">The architecture of the neural waveshaping unit</div>

The NEWT also learns to shift and scale the input and output of the shaping function in response to a control signal.
This allows a trained NEWT to ''play'' the waveshaper just like the distortion index $$a[n]$$ and normalising coefficient $$N[n]$$ in the waveshaping synthesiser we defined above.
In practice we use multiple NEWTs in parallel -- 64, to be precise -- and they share an MLP to generate these affine parameters from the control signal.

So what does the NEWT actually learn?
How do we know that the NEWTs aren't just learning to attenuate the harmonic exciter signal?
Well, we can visualise the shaping functions learnt by our model by simply sampling across their domain.
The following figure plots the shaping functions learnt by the sinusoidal MLPs in each NEWT in our violin model sampled across the interval $$[-3, 3]$$.

<div class="row">
    <div style="width: 10%">
    </div>
    <div class="col-sm mt-3 mt-md-0">
        <img class="img-fluid rounded " src="{{ '/assets/img/newt_shapers.png' | relative_url }}" alt="" title="Shaping functions learnt by the NEWT"/>
    </div>
    <div style="width: 10%">
    </div>
</div>
<div class="caption">All 64 shaping functions learned by the NEWT bank in our violin model</div>

Clearly, the NEWTs learn a wide variety of shaping functions from the data.

But couldn't the model simply shift and scale the exciter signal to only utilise a locally linear region of the shaping function?
This seems like a very real possibility!
So, let's inspect the affine parameters generated by our model whilst performing timbre transfer on a ~20 second long audio clip (the first row in our timbre transfer examples below).
This next figure plots these parameters for *every* one of the 64 NEWTs, again using the violin model.

<div class="row" style="width: 100%; margin-left:0%;">
    <div style="width: 0%">
    </div>
    <div class="col-sm mt-3 mt-md-0">
        <img class="img-fluid rounded" src="{{ '/assets/img/shaper_coeffs.png' | relative_url }}" alt="" title="Affine transform parameters generated by the NEWT to synthesise audio"/>
    </div>
    <div style="width: 0%">
    </div>
</div>
<div class="caption">Affine transform parameters generated by the NEWT to synthesise audio</div>

The model clearly makes use of $$\alpha_a$$, $$\beta_a$$, and $$\alpha_N$$, varying many of these parameters widely across the length of the audio clip.
However, $$\beta_N$$ is barely varied at all.
This tells us that it has learnt to continuously shift and scale the input to the shaper functions, and also scale the output, whilst leaving the DC offset of the output signal almost untouched.
It is also interesting to note that many affine parameters seem highly correlated, whilst a few appear to move independently, suggesting that some NEWTs may work together to produce cohesive aspects of the signal, whilst others take momentary individual responsibility for signal components.

Whilst it's possible to imagine what effect the $$\alpha_N$$ affine parameter might have on a signal — it simply changes the amplitude — it's harder to get a sense of how $$\alpha_a$$ and $$\beta_a$$ might change things.
Do they result in any meaningful distortion?
As the NEWT operates at audio rate, we can simply listen to the output of the shaping function to find out!
Here, we have randomly selected a NEWT (index 26 in our violin model), and used the corresponding $$\alpha_a$$ and $$\beta_a$$ plotted above.
To make the effect of the NEWT clear, we have simply fed it with a steady sine wave at 250 Hz, so all the variation you hear in the sound is caused by the affine parameters and shaping function.
On the left, you can listen to the sine wave input, and on the right you can listen to what the NEWT does to this signal!
You might want to turn your speakers down before listening -- these are not pleasant sounds.


<div class="row">
    <div style="width: 50%; text-align: center;">
        <h4>input</h4>
        <audio controls>
        <source src="{{ '/assets/audio/shaper_26_in.wav' | relative_url }}" type="audio/wav" />
        </audio>
    </div>
    <div style="width: 50%; text-align: center;">
        <h4>output</h4>
        <audio controls>
        <source src="{{ '/assets/audio/shaper_26_out.wav' | relative_url }}" type="audio/wav" />
        </audio>
    </div>
</div>
<div class="caption">A sine wave at 250 Hz before and after being passed through a randomly selected NEWT shaper, with control signals extracted from another audio file</div>

As you can hear, this shaper combined with the affine parameters generated from real control signals results in very distinct changes to the harmonic content of the exciter signal.
This tells us that the NEWT does indeed learn to manipulate the harmonic content of the exciter signal.


### bringing it all together

We use the bank of 64 NEWTs in the context of a differentiable harmonic-plus-noise synthesiser, very much like the one used in the DDSP autoencoder[^1].
In our model, however, rather than directly generating harmonic amplitudes with a recurrent decoder, the NEWTs learn shaping functions which encode particular harmonic profiles and evolutions.
The sequence model then generates affine transform parameters which utilise these shapers to generate timbres.

As well as the NEWT bank, our model contains two further differentiable signal processors: a learnable convolution reverb, and a filtered noise synthesiser which contributes aperiodic components to the final signal.

<div class="row">
    <div style="width: 1%">
    </div>
    <div class="col-sm mt-3 mt-md-0">
        <img class="img-fluid rounded " src="{{ '/assets/img/nws.png' | relative_url }}" alt="" title="The architecture of our synthesis model"/>
    </div>
    <div style="width: 1%">
    </div>
</div>
<div class="caption">The full neural waveshaping synthesis model</div>

This architecture uses just 266,000 trainable parameters, and learns from just minutes of unlabelled audio!
As with other recent models in the DDSP family, such as Engel, et al.'s differentiable additive synthesiser[^1] and Wang, et al.'s differentiable source-filter model[^4], our model exploits a strong inductive bias towards the generation of periodic signals with a harmonic structure.


### why this approach?

You may be wondering why, with so many successful recent approaches to monophonic neural audio synthesis, would you choose to use our proposed method?
Bluntly, the answer is efficiency.

Real-time performance is critical for practical use in sound design and music production applications, and existing models are typically very large.
Some do just about manage to achieve real-time performance (e.g. see Antoine Caillon's excellent [real-time implementation of the DDSP autoencoder](https://github.com/acids-ircam/ddsp_pytorch)), but this typically requires big compromises in the model architecture, a GPU, and/or impractical buffer sizes.
Our approach, on the other hand, comfortably achieves real-time performance at a number of reasonable buffer sizes on both GPU *and* CPU.

<div class="row">
    <div style="width: 20%">
    </div>
    <div class="col-sm mt-3 mt-md-0">
        <img class="img-fluid rounded " src="{{ '/assets/img/rtf_buffer.png' | relative_url }}" alt="" title="The mean real-time factor of our model and benchmarks at a number of buffer sizes"/>
    </div>
    <div style="width: 20%">
    </div>
</div>
<div class="caption">The mean real-time factor of our model and benchmarks at a number of buffer sizes</div>


How does it do this?
Our baseline NEWT implementation takes advantage of the ability to parallelise grouped 1D convolutions across a sequence.
This alone allows very efficient generation, but we further improve CPU performance with an optimisation called the FastNEWT.
In brief, this involves sampling the domain of the NEWT's shaping function and replacing it with a lookup table, thus replacing the forward pass through each sinusoidal MLP with the $$\mathcal{O}(1)$$ operation of an array lookup.
In listening tests and quantitative evaluation, we found this optimisation has no significant effect on synthesis quality!

<!-- ### real-time performance

**You** might wonder what the advantage of this method is. -->

<hr />

### audio examples

So how does it sound?
We have collected a variety of audio examples here to help you make up your own mind!

<div style="text-align: center; width=60%; margin:auto;">
    <h4>resynthesis</h4>
    <p>
        In these examples, we extracted F0 and loudness control signals from the corresponding test dataset.
        We then fed these through the model to generate audio.
        We also provide outputs from a DDSP-tiny model trained on the same data for comparison.
    </p>
</div>

<br />
<div style="text-align: center">
<h4>flute</h4>
</div>
<div class="row" style="width: 100%; margin-left: 0%">
    <div style="width: 25%; text-align: center;">
        <h4>reference</h4>
        <audio controls style="width: 90%">
        <source src="{{ '/assets/audio/nws_examples/fl1/fl1_ref.wav' | relative_url }}" type="audio/wav" />
        </audio>
    </div>
    <div style="width: 25%; text-align: center;">
        <h4>ours</h4>
        <audio controls style="width: 90%">
        <source src="{{ '/assets/audio/nws_examples/fl1/fl1_nws.wav' | relative_url }}" type="audio/wav" />
        </audio>
    </div>
    <div style="width: 25%; text-align: center;">
        <h4>ours + FastNEWT</h4>
        <audio controls style="width: 90%">
        <source src="{{ '/assets/audio/nws_examples/fl1/fl1_nws_fn.wav' | relative_url }}" type="audio/wav" />
        </audio>
    </div>
    <div style="width: 25%; text-align: center;">
        <h4>DDSP-tiny</h4>
        <audio controls style="width: 90%">
        <source src="{{ '/assets/audio/nws_examples/fl1/fl1_ddsp_tiny.wav' | relative_url }}" type="audio/wav" />
        </audio>
    </div>
</div>
<div class="row" style="width: 100%; margin-left:0%;">
    <div style="width: 25%; text-align: center;">
        <audio controls style="width: 90%">
        <source src="{{ '/assets/audio/nws_examples/fl2/fl2_ref.wav' | relative_url }}" type="audio/wav" />
        </audio>
    </div>
    <div style="width: 25%; text-align: center;">
        <audio controls style="width: 90%">
        <source src="{{ '/assets/audio/nws_examples/fl2/fl2_nws.wav' | relative_url }}" type="audio/wav" />
        </audio>
    </div>
    <div style="width: 25%; text-align: center;">
        <audio controls style="width: 90%">
        <source src="{{ '/assets/audio/nws_examples/fl2/fl2_nws_fn.wav' | relative_url }}" type="audio/wav" />
        </audio>
    </div>
    <div style="width: 25%; text-align: center;">
        <audio controls style="width: 90%">
        <source src="{{ '/assets/audio/nws_examples/fl2/fl2_ddsp_tiny.wav' | relative_url }}" type="audio/wav" />
        </audio>
    </div>
</div>
<div class="row" style="width: 100%; margin-left:0%;">
    <div style="width: 25%; text-align: center;">
        <audio controls style="width: 90%">
        <source src="{{ '/assets/audio/nws_examples/fl3/fl1_dm_cond.wav' | relative_url }}" type="audio/wav" />
        </audio>
    </div>
    <div style="width: 25%; text-align: center;">
        <audio controls style="width: 90%">
        <source src="{{ '/assets/audio/nws_examples/fl3/fl1_dm_nws.wav' | relative_url }}" type="audio/wav" />
        </audio>
    </div>
    <div style="width: 25%; text-align: center;">
        <audio controls style="width: 90%">
        <source src="{{ '/assets/audio/nws_examples/fl3/fl1_dm_nws_fn.wav' | relative_url }}" type="audio/wav" />
        </audio>
    </div>
    <div style="width: 25%; text-align: center;">
        <audio controls style="width: 90%">
        <source src="{{ '/assets/audio/nws_examples/fl3/fl1_dm_ddsp_tiny.wav' | relative_url }}" type="audio/wav" />
        </audio>
    </div>
</div>
<br />
<div style="text-align: center">
<h4>trumpet</h4>
</div>
<div class="row" style="width: 100%; margin-left:0%;">
    <div style="width: 25%; text-align: center;">
        <h4>reference</h4>
        <audio controls style="width: 90%">
        <source src="{{ '/assets/audio/nws_examples/tpt1/tpt1_ref.wav' | relative_url }}" type="audio/wav" />
        </audio>
    </div>
    <div style="width: 25%; text-align: center;">
        <h4>ours</h4>
        <audio controls style="width: 90%">
        <source src="{{ '/assets/audio/nws_examples/tpt1/tpt1_nws.wav' | relative_url }}" type="audio/wav" />
        </audio>
    </div>
    <div style="width: 25%; text-align: center;">
        <h4>ours + FastNEWT</h4>
        <audio controls style="width: 90%">
        <source src="{{ '/assets/audio/nws_examples/tpt1/tpt1_nws_fn.wav' | relative_url }}" type="audio/wav" />
        </audio>
    </div>
    <div style="width: 25%; text-align: center;">
        <h4>DDSP-tiny</h4>
        <audio controls style="width: 90%">
        <source src="{{ '/assets/audio/nws_examples/tpt1/tpt1_ddsp_tiny.wav' | relative_url }}" type="audio/wav" />
        </audio>
    </div>
</div>
<div class="row" style="width: 100%; margin-left:0%;">
    <div style="width: 25%; text-align: center;">
        <audio controls style="width: 90%">
        <source src="{{ '/assets/audio/nws_examples/tpt2/tpt2_ref.wav' | relative_url }}" type="audio/wav" />
        </audio>
    </div>
    <div style="width: 25%; text-align: center;">
        <audio controls style="width: 90%">
        <source src="{{ '/assets/audio/nws_examples/tpt2/tpt2_nws.wav' | relative_url }}" type="audio/wav" />
        </audio>
    </div>
    <div style="width: 25%; text-align: center;">
        <audio controls style="width: 90%">
        <source src="{{ '/assets/audio/nws_examples/tpt2/tpt2_nws_fn.wav' | relative_url }}" type="audio/wav" />
        </audio>
    </div>
    <div style="width: 25%; text-align: center;">
        <audio controls style="width: 90%">
        <source src="{{ '/assets/audio/nws_examples/tpt2/tpt2_ddsp_tiny.wav' | relative_url }}" type="audio/wav" />
        </audio>
    </div>
</div>
<div class="row" style="width: 100%; margin-left:0%;">
    <div style="width: 25%; text-align: center;">
        <audio controls style="width: 90%">
        <source src="{{ '/assets/audio/nws_examples/tpt3/tpt1_dm_cond.wav' | relative_url }}" type="audio/wav" />
        </audio>
    </div>
    <div style="width: 25%; text-align: center;">
        <audio controls style="width: 90%">
        <source src="{{ '/assets/audio/nws_examples/tpt3/tpt1_dm_nws.wav' | relative_url }}" type="audio/wav" />
        </audio>
    </div>
    <div style="width: 25%; text-align: center;">
        <audio controls style="width: 90%">
        <source src="{{ '/assets/audio/nws_examples/tpt3/tpt1_dm_nws_fn.wav' | relative_url }}" type="audio/wav" />
        </audio>
    </div>
    <div style="width: 25%; text-align: center;">
        <audio controls style="width: 90%">
        <source src="{{ '/assets/audio/nws_examples/tpt3/tpt1_dm_ddsp_tiny.wav' | relative_url }}" type="audio/wav" />
        </audio>
    </div>
</div>
<br />
<div style="text-align: center">
<h4>violin</h4>
</div>
<div class="row" style="width: 100%; margin-left:0%;">
    <div style="width: 25%; text-align: center;">
        <h4>reference</h4>
        <audio controls style="width: 90%">
        <source src="{{ '/assets/audio/nws_examples/vn1/vn1_ref.wav' | relative_url }}" type="audio/wav" />
        </audio>
    </div>
    <div style="width: 25%; text-align: center;">
        <h4>ours</h4>
        <audio controls style="width: 90%">
        <source src="{{ '/assets/audio/nws_examples/vn1/vn1_nws.wav' | relative_url }}" type="audio/wav" />
        </audio>
    </div>
    <div style="width: 25%; text-align: center;">
        <h4>ours + FastNEWT</h4>
        <audio controls style="width: 90%">
        <source src="{{ '/assets/audio/nws_examples/vn1/vn1_nws_fn.wav' | relative_url }}" type="audio/wav" />
        </audio>
    </div>
    <div style="width: 25%; text-align: center;">
        <h4>DDSP-tiny</h4>
        <audio controls style="width: 90%">
        <source src="{{ '/assets/audio/nws_examples/vn1/vn1_ddsp_tiny.wav' | relative_url }}" type="audio/wav" />
        </audio>
    </div>
</div>
<div class="row" style="width: 100%; margin-left:0%;">
    <div style="width: 25%; text-align: center;">
        <audio controls style="width: 90%">
        <source src="{{ '/assets/audio/nws_examples/vn2/vn2_ref.wav' | relative_url }}" type="audio/wav" />
        </audio>
    </div>
    <div style="width: 25%; text-align: center;">
        <audio controls style="width: 90%">
        <source src="{{ '/assets/audio/nws_examples/vn2/vn2_nws.wav' | relative_url }}" type="audio/wav" />
        </audio>
    </div>
    <div style="width: 25%; text-align: center;">
        <audio controls style="width: 90%">
        <source src="{{ '/assets/audio/nws_examples/vn2/vn2_nws_fn.wav' | relative_url }}" type="audio/wav" />
        </audio>
    </div>
    <div style="width: 25%; text-align: center;">
        <audio controls style="width: 90%">
        <source src="{{ '/assets/audio/nws_examples/vn2/vn2_ddsp_tiny.wav' | relative_url }}" type="audio/wav" />
        </audio>
    </div>
</div>
<div class="row" style="width: 100%; margin-left:0%;">
    <div style="width: 25%; text-align: center;">
        <audio controls style="width: 90%">
        <source src="{{ '/assets/audio/nws_examples/vn3/vn1_dm_cond.wav' | relative_url }}" type="audio/wav" />
        </audio>
    </div>
    <div style="width: 25%; text-align: center;">
        <audio controls style="width: 90%">
        <source src="{{ '/assets/audio/nws_examples/vn3/vn1_dm_nws.wav' | relative_url }}" type="audio/wav" />
        </audio>
    </div>
    <div style="width: 25%; text-align: center;">
        <audio controls style="width: 90%">
        <source src="{{ '/assets/audio/nws_examples/vn3/vn1_dm_nws_fn.wav' | relative_url }}" type="audio/wav" />
        </audio>
    </div>
    <div style="width: 25%; text-align: center;">
        <audio controls style="width: 90%">
        <source src="{{ '/assets/audio/nws_examples/vn3/vn1_dm_ddsp_tiny.wav' | relative_url }}" type="audio/wav" />
        </audio>
    </div>
</div>
<br />
<hr />
<br />

<div style="text-align: center; width=60%; margin:auto;">
    <h4>timbre transfer</h4>
    <p>
        In these examples, we extracted F0 and loudness control signals from monophonic audio from an entirely different timbral domain -- in this case, vocal stems.
        We then fed these through the model to generate audio which retains the pitch and loudness from the source audio whilst the timbre is generated by the model.
    </p>
</div>

<div class="row" style="width: 100%; margin-left:0%;">
    <div style="width: 25%; text-align: center;">
        <h4>source</h4>
        <audio controls style="width: 90%">
        <source src="{{ '/assets/audio/nws_examples/tt/tt1_in.wav' | relative_url }}" type="audio/wav" />
        </audio>
    </div>
    <div style="width: 25%; text-align: center;">
        <h4>flute</h4>
        <audio controls style="width: 90%">
        <source src="{{ '/assets/audio/nws_examples/tt/tt1_fl.wav' | relative_url }}" type="audio/wav" />
        </audio>
    </div>
    <div style="width: 25%; text-align: center;">
        <h4>trumpet</h4>
        <audio controls style="width: 90%">
        <source src="{{ '/assets/audio/nws_examples/tt/tt1_tpt.wav' | relative_url }}" type="audio/wav" />
        </audio>
    </div>
    <div style="width: 25%; text-align: center;">
        <h4>violin</h4>
        <audio controls style="width: 90%">
        <source src="{{ '/assets/audio/nws_examples/tt/tt1_vn.wav' | relative_url }}" type="audio/wav" />
        </audio>
    </div>
</div>
<div class="row" style="width: 100%; margin-left:0%;">
    <div style="width: 25%; text-align: center;">
        <audio controls style="width: 90%">
        <source src="{{ '/assets/audio/nws_examples/tt/tt2_in.wav' | relative_url }}" type="audio/wav" />
        </audio>
    </div>
    <div style="width: 25%; text-align: center;">
        <audio controls style="width: 90%">
        <source src="{{ '/assets/audio/nws_examples/tt/tt2_fl.wav' | relative_url }}" type="audio/wav" />
        </audio>
    </div>
    <div style="width: 25%; text-align: center;">
        <audio controls style="width: 90%">
        <source src="{{ '/assets/audio/nws_examples/tt/tt2_tpt.wav' | relative_url }}" type="audio/wav" />
        </audio>
    </div>
    <div style="width: 25%; text-align: center;">
        <audio controls style="width: 90%">
        <source src="{{ '/assets/audio/nws_examples/tt/tt2_vn.wav' | relative_url }}" type="audio/wav" />
        </audio>
    </div>
</div>
<div class="row" style="width: 100%; margin-left:0%;">
    <div style="width: 25%; text-align: center;">
        <audio controls style="width: 90%">
        <source src="{{ '/assets/audio/nws_examples/tt/tt3_in.wav' | relative_url }}" type="audio/wav" />
        </audio>
    </div>
    <div style="width: 25%; text-align: center;">
        <audio controls style="width: 90%">
        <source src="{{ '/assets/audio/nws_examples/tt/tt3_fl.wav' | relative_url }}" type="audio/wav" />
        </audio>
    </div>
    <div style="width: 25%; text-align: center;">
        <audio controls style="width: 90%">
        <source src="{{ '/assets/audio/nws_examples/tt/tt3_tpt.wav' | relative_url }}" type="audio/wav" />
        </audio>
    </div>
    <div style="width: 25%; text-align: center;">
        <audio controls style="width: 90%">
        <source src="{{ '/assets/audio/nws_examples/tt/tt3_vn.wav' | relative_url }}" type="audio/wav" />
        </audio>
    </div>
</div>

<br />
<hr />
<br />






### references

[^1]: J. Engel, L. H. Hantrakul, C. Gu, and A. Roberts, “DDSP: Differentiable Digital Signal Processing,” in *8th International Conference on Learning Representations*, Addis Ababa, Ethiopia, 2020.
[^2]: M. Le Brun, “Digital Waveshaping Synthesis,” *Journal of the Audio Engineering Society*, vol. 27, no. 4, pp.  250–266, Apr. 1979.
[^3]: V. Sitzmann, J. N. P. Martel, A. W. Bergman, D. B. Lindell, and G. Wetzstein, “Implicit Neural Representations with Periodic Activation Functions,” *arXiv:2006.09661 [cs, eess]*, Jun. 2020.
[^4]: X. Wang, S. Takaki, and J. Yamagishi, “Neural Source-Filter Waveform Models for Statistical Parametric Speech Synthesis,” *IEEE/ACM Transactions on Audio, Speech, and Language Processing*, vol. 28, pp. 402–415, 2020.