Simple Codec
===========

A Simple Perceptual Audio Coder (and decoder)

## ABSTRACT

A perceptual audio coder was written for Matlab which applies simple masking curves to determine the proper bit allocation for the quantization of MDCT coefficients.  This coder was determined to be transparent at 256kbps and nearly transparent at 128kbps, depending on material.  The perceptual coder was also compared to an LSE coder and was found to be significantly worse.  Several of the artifacts that deteriorate the quality of the perceptual coder are well known, and will be fixed in upcoming versions of this coder.

## Table of Contents

[Introduction](#introduction)

- [Audio Coding](#audio-coding)

- [Masking](#masking)

[Design](#design)

- [Frames](#frames)

- [Signal-to-Mask Ratio](#signal-to-mask-ratio)

- [Bit Allocation](#bit-allocation)

- [Quantization](#quantization)

- [Writing the file](#writing-the-file)

[Subjective Testing](#subjective-testing)

- [Subjective Difference Grade](#subjective-difference-grade)

- [Comparison to LSE Coder](#comparison-to-lse-coder)

[Conclusions](#conclusions)

- [Current Coder](#current-coder)

- [Future Improvements](#future-improvements)

[APPENDIX](#appendix)

[RESOURCES](#resources)

## INTRODUCTION

### Audio Coding

Over the past decade, audio coding has become a hot topic due to the popularity of internet media, and improvements in movie technology.  Various methods have been proposed to improve low bit-rate audio quality.  Some coders, such as Meridian Lossless Packing, WaveArc, Pegasus SPS, Sonarc, WavPack, AudioZip, Monkey, and FLAC are considered &#8220;lossless&#8221; because they retain all the original audio data while reducing bit-rate.  Others coders are called &#8220;lossy&#8221; because they throw away portions of the audio stream that cannot be easily heard, and many are commonly in use today including MP3, WMA, AAC, PAC. Such audio coders are called perceptual audio coders, because they rely on human perception of sound.

### Masking

Many portions of an audio stream cannot actually be heard.  Any sound with intensity below a certain threshold (called the threshold in quiet) cannot be heard, due to the limits of the ear&#8217;s sensitivity.  Sometimes, sounds above the threshold in quiet cannot be heard because other sounds cover them up.  This is due to a psychoacoustic phenomenon known as masking. If two separate tones are close enough in frequency, one tone may actually cover up the other one.  The tone that is heard is called the masker, and the tone which is not heard is called the maskee.

![](http://www.perceptualentropy.com/images/masking1.jpg)

The above phenomenon is known as simultaneous masking.  There is another phenomenon known as temporal masking.  Temporal
masking is the masking of a sound before or after the masker event occurs.

![](http://www.perceptualentropy.com/images/masking2.jpg)

## DESIGN

### Frames

The first step in designing an audio coder is to segment the audio stream into frames.  A frame is a short section of audio, typically less than 50ms each.  At a sampling frequency of 44.1kHz, a frame of 2048 samples is about 46ms long.  Enframing the audio stream allows the engineer to treat each frame as a relatively stationary sound.  Frame lengths longer than 50ms are typically not used, since pleasant sounding audio is non-stationary. Large frame sizes are good for audio signals that change relatively slowly.  Longer frame lengths also allow greater frequency resolution, because the FFT length is also greater.  Shorter frame sizes may be used for more transient sounds, to avoid an artifact known as pre-echo.  Pre-echo occurs when a transient sound is averaged over a long period of time (such as 46ms), and causes an attack to sound blurred. Variable bit-rate coders take advantage of both long and short frame lengths by changing the frame length dynamically, depending on the material.  The coder presented in this paper uses a fixed frame length of 2048 samples.

### Signal-to-Mask Ratio

There are many ways to calculate the masking threshold.  In general, the masking threshold varies depending on the frequency and intensity of the masker signal.  The width of the masking curve typically extends farther in the direction of higher frequencies than toward lower frequencies, and the amplitude depends on both the frequency and the tonality of the masker. Noise tends to mask much more than tones do.

In order to calculate the masking threshold, the first step is to calculate the FFT of the frame, and find the spectral
peaks.  To find the peaks, simply search for every point where the slope changes from positive to negative.  Each of these peaks corresponds to individual frequencies in the signal.  The next step is to calculate the approximate SPL of each peak. One way to do this is to normalize such that a full-amplitude tone at 1kHz is equal to 96dB.  The SPL of each peak can then be easily calculated according to the following formula:

![](http://www.perceptualentropy.com/images/equation1.jpg)

From these SPL values, a masking curve can be created for each peak.  There are several masking functions that do this.
This coder uses the function suggested by Schroeder:

![](http://www.perceptualentropy.com/images/equation2.jpg)

where dz is the bark frequency, defined by the following equation:

![](http://www.perceptualentropy.com/images/equation3.jpg)

where f is the frequency in kHz.

The next step is to combine all the masking curves with the Threshold in Quiet (TIQ).  The TIQ is the minimum SPL that a
person can hear at a given frequency, and is typically defined by the following equation:

![](http://www.perceptualentropy.com/images/equation4.jpg)

where f is the frequency in kHz.

There are a variety of ways to combine the masking curves and the TIQ which correspond to different values of alpha in the equation

![](http://www.perceptualentropy.com/images/equation5.jpg)

In this coder, alpha is zero, which corresponds to using the highest masking curve (or the threshold in quiet).
The signal-to-mask ratio (SMR) can easily be calculated by dividing the SPL of the signal by the masking threshold.

### Bit Allocation

From the SMR, it can determined which frequency bands should receive the most bits.  As a general rule, each bit increases signal-to-noise ratio by about 6dB.  Therefore, allocating a bit for each 6dB of SMR would ensure that quantization noise is below the masking threshold, and thus inaudible.  However, there may not be enough bits available to do this, bits must be allocated to where they are needed most.  The &#8220;water-filling&#8221; bit allocation algorithm is used to allocate bits by looking for the maximum value of the SMR, allocating a bit to that subband, subtracting 6dB from the SMR at that frequency, and repeating as long as bits are available to allocate.

### Quantization

After determining where bits should be allocated, the next step is to quantize the audio signal to the appropriate number of bits.  This audio coder is based on the Modified Discrete Cosine Transform (MDCT), so the MDCT coefficients are quantized. The MDCT of the original time-domain frame must first be computed.  Then the coefficients must be attenuated because values as large as those typically found in the MDCT cannot typically be quantized.  Therefore, an attenuation factor is chosen equal to the maximum value found in the MDCT, reducing the maximum value that needs to be quantized to unity.  After attenuating the coefficients, they are quantized according to the bit allocation scheme determined earlier.

### Reading/Writing the files

Once the MDCT coefficients are quantized, they can be written to a file.  In addition to the MDCT coefficients, the gain
factor must also be specified as well as the number of bits allocated to each band.  In this coder, a file header is also
included which contains information such as the sampling frequency, frame length, bit rate, number of bits used for writing the gain factor, and the number of frames in the file.  Because only a few bits are to be used to represent the gain factor, the logarithm of the gain is written to the file.

![](http://www.perceptualentropy.com/images/flowchart1.jpg)

![](http://www.perceptualentropy.com/images/flowchart2.jpg)

## SUBJECTIVE TESTING

### Subjective Difference Grade

A standard means of conducting subjective listening tests is to have a variety of listeners listen to various sounds and
rate them on a scale of 1-5.

![](http://www.perceptualentropy.com/images/SDG.jpg)

The subjective difference grade is determined by subtracting the grade of the reference signal from the grade of the coded signal.  The perceptual coder was used to encode three difference sounds- a flute, drums, and speech.  The results of subjective tests for the three signals are shown below.

![](http://www.perceptualentropy.com/images/SDGrades.jpg)

### Comparison to LSE Coder

An alternative audio coder was designed to allocate bits based simply on energy, thus producing an audio signal with the
Least Squared Error (LSE).  The results of subjective testing of this LSE coder are compared to those of the perceptual
coder in the following graph:

![](http://www.perceptualentropy.com/images/SDG2.jpg)

## CONCLUSIONS

### Current Coder

As a simple audio coder, this algorithm appears to work fairly well.  However, as a perceptual audio coder, it does not
perform well at all.  It is greatly inferior to the LSE coder, suggesting that the perceptual model needs improvement.
The first problem is pre-echo.  Although pre-echo is not the most annoying artifact, it is present due to the large frame
length.  There is another artifact present known as &#8220;birdies&#8221; which adds a flying-saucer type sound.  This is created when the masking thresholds change slightly over time, causing drastic changes in bit allocation.  Each of these artifacts caused by the perceptual coder can be minimized by adjusting certain things in the coder.  Pre-echo can be minimized by allowing variable frame lengths.  It may also be possible to minimize birdies by enhancing the perceptual model.

### Future Improvements

Next semester, I plan to improve this coder by adding a number of advanced functionalities and improving the existing code. The first change I plan to make concerns the perceptual model.  The current model is based on a single function that only varies slightly with frequency.  I plan to do some research to determine the best perceptual models to employ, and to implement masking functions that vary with frequency, loudness, and tonality.  The next change I will make is to add variable bit-rate capability.  This will allow the frames to change to a small block size when transients are present, and switch back to long frame lengths when the sound is relatively constant.  This will decrease pre-echo and speech
reverberation while maintaining good frequency resolution when needed.  I also plan to incorporate stereo coding into the
next version of this coder.  This will allow stereo files to be encoded more efficiently, while allowing me to learn about binaural masking.

## APPENDIX

Encoded Sounds: ([Click Here to Downloads These Sounds](http://www.perceptualentropy.com/sounds/sounds.zip))

|Original|128kbps|64kbps|32kbps|
|--------|-------|------|------|
|[Flute](http://www.perceptualentropy.com/sounds/fluteA.wav)|[Flute - Perceptual](http://www.perceptualentropy.com/sounds/fluteB.wav)|[Flute - Perceptual](http://www.perceptualentropy.com/sounds/fluteD.wav)|[Flute - Perceptual](http://www.perceptualentropy.com/sounds/fluteF.wav)|
||[Flute - Spectral Power](http://www.perceptualentropy.com/sounds/fluteC.wav)|[Flute - Spectral Power](http://www.perceptualentropy.com/sounds/fluteE.wav)|[Flute - Spectral Power](http://www.perceptualentropy.com/sounds/fluteG.wav)|
|[Drums](http://www.perceptualentropy.com/sounds/drumsA.wav)|[Drums - Perceptual](http://www.perceptualentropy.com/sounds/drumsB.wav)|[Drums - Perceptual](http://www.perceptualentropy.com/sounds/drumsD.wav)|[Drums - Perceptual](http://www.perceptualentropy.com/sounds/drumsF.wav)|
||[Drums - Spectral Power](http://www.perceptualentropy.com/sounds/drumsC.wav)|[Drums - Spectral Power](http://www.perceptualentropy.com/sounds/drumsE.wav)|[Drums - Spectral Power](http://www.perceptualentropy.com/sounds/drumsG.wav)|
|[Speech](http://www.perceptualentropy.com/sounds/speechA.wav)|[Drums - Perceptual](http://www.perceptualentropy.com/sounds/drumsB.wav)|[Drums - Perceptual](http://www.perceptualentropy.com/sounds/drumsD.wav)|[Drums - Perceptual](http://www.perceptualentropy.com/sounds/drumsF.wav)|
||[Speech - Spectral Power](http://www.perceptualentropy.com/sounds/speechC.wav)|[Speech - Spectral Power](http://www.perceptualentropy.com/sounds/speechE.wav)|[Speech - Spectral Power](http://www.perceptualentropy.com/sounds/speechG.wav)|


## RESOURCES

M. Bosi, R. Goldberg, Introduction to Digital Audio Coding and Standards, Kluwer Academic Publishers, 2003.

Audio Engineering Society CD-ROM, &#8220;Perceptual Audio Coders: What to Listen For&#8221;, AES 2001.

&copy; Jon Boley [http://www.jboley.com](http://www.jboley.com/)

