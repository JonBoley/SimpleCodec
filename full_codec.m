function Fs = new_codec()
% yourfile.wav is the input file
% yourfile.jon is the encoded file
% decoded_yourfile.wav is the decoded output file
%
% If you make any modifications to this code, I would
% like to hear about it.
% - Jon Boley (jdb@jboley.com)

clear all;

scalebits = 4;
bitrate = 128000;
N = 2048; % framelength
original_filename = sprintf('yourfile.wav');
coded_filename = sprintf('yourfile.jon');
decoded_filename = sprintf('decoded_yourfile.wav');

[Y,Fs,NBITS] = wavread(original_filename);
tone = Y;

num_subbands = floor(fftbark(N/2,N/2,Fs))+1;
bits_per_frame = floor(((bitrate/Fs)*(N/2)) - (scalebits*num_subbands));


sig=sin(2*pi*1000*[1/Fs:1/Fs:(N/2)/Fs]);
win=(0.5 - 0.5*cos((2*pi*([1:(N/2)]-0.5))/(N/2))); 
fftmax = max(abs(fft(sig.*win))); % defined as 96dB

%   Enframe Audio   
FRAMES = enframe(tone,N,N/2);

% Write File Header 
fid = fopen(coded_filename,'w');
fwrite(fid, Fs, 'ubit16'); % Sampling Frequency
fwrite(fid, N, 'ubit12'); % Frame Length
fwrite(fid, bitrate, 'ubit18'); % Bit Rate
fwrite(fid, scalebits, 'ubit4'); % Number of Scale Bits per Sub-Band
fwrite(fid, length(FRAMES(:,1)), 'ubit26'); % Number of frames
    
%   Computations    
for frame_count=1:length(FRAMES(:,1))

    if mod(frame_count,10) == 0
        outstring = sprintf('Now Encoding Frame %i of %i', frame_count, length(FRAMES(:,1)));
        disp(outstring);
    end
    
    fft_frame = fft(FRAMES(frame_count,:));

    if fft_frame == zeros(1,N)
        Gain = zeros(1,floor(fftbark(N/2,N/2,Fs))+1);
        bit_alloc = zeros(1,floor(fftbark(N/2,N/2,Fs))+1);
    else    
        len = length(fft_frame);
        peak_width = zeros(1,len);
        peak_points = cell(len,len);
        peak_min_value = zeros(1,len);

        % Find Peaks
        centers = find(diff(sign(diff( abs(fft_frame).^2) )) == -2) + 1;
        spectral_density = zeros(1,length(centers));
    
        for k=1:length(centers)
            peak_max(k) = centers(k) +2;
            peak_min(k) = centers(k) - 2;
            peak_width(k) = peak_max(k) - peak_min(k);
            
            for j=peak_min(k):peak_max(k)
                if (j > 0) & (j < N)
                    spectral_density(k) = spectral_density(k) + abs(fft_frame(j))^2;
                end
            end
        end
            
        % This gives the amplitude squared of the original signal
        modified_SD = spectral_density / ((N^2)/8);
        SPL = 96 + 10*log10(modified_SD);
        
        % TRANSFORM FFT'S TO SPL VALUES
        fft_spl = 96 + 20*log10(abs(fft_frame)/fftmax);
    
    
        % Threshold in Quiet
        f_kHz = [1:Fs/N:Fs/2];
        f_kHz = f_kHz/1000;
        A = 3.64*(f_kHz).^(-0.8) - 6.5*exp(-0.6*(f_kHz - 3.3).^2) + (10^(-3))*(f_kHz).^4;
    
        % Masking Spectrum
        big_mask = max(A,Schroeder(centers(1)*(Fs/2)/N,fft_spl(centers(1)),...
            14.5+bark(centers(1)*(Fs/2)/N)));
        for peak_count=2:length(centers)
            try
            big_mask = max(big_mask,Schroeder(centers(peak_count)*(Fs/2)/N,fft_spl((peak_count)),...
                14.5+bark(centers(peak_count)*(Fs/2)/N)));
            catch
                peak_count=peak_count;
            end
        end

        % Signal Spectrum - Masking Spectrum (with max of 0dB)
        New_FFT = fft_spl(1:N/2)-big_mask;
        New_FFT_indices = find(New_FFT > 0);
        New_FFT2 = zeros(1,N/2);
        for i=1:length(New_FFT_indices)
            New_FFT2(New_FFT_indices(i)) = New_FFT(New_FFT_indices(i));
        end
    
        if frame_count == 55
            semilogx([0:(Fs/2)/(N/2):Fs/2-1],fft_spl(1:N/2),'b');
            hold on;
            semilogx([0:(Fs/2)/(N/2):Fs/2-1],big_mask,'m');
            hold off;
            title('Signal (blue) and Masking Spectrum (pink)');
            figure;
            semilogx([0:(Fs/2)/(N/2):Fs/2-1],New_FFT2);
            title('SMR');
            figure;
            stem(allocate(New_FFT2,bits_per_frame,N,Fs));
            title('Bits perceptually allocated');
        end
    
        bit_alloc = allocate(New_FFT2,bits_per_frame,N,Fs);
    
        [Gain,Data] = p_encode(mdct(FRAMES(frame_count,:)),Fs,N,bit_alloc,scalebits);
    end % end of If-Else Statement        
        
    % Write Audio Data to File
    qbits = sprintf('ubit%i', scalebits);
    fwrite(fid, Gain, qbits);
    fwrite(fid, bit_alloc, 'ubit4');
    for i=1:25
        indices = find((floor(fftbark([1:N/2],N/2,Fs))+1)==i);
        qbits = sprintf('ubit%i', bit_alloc(i)); % bits(floor(fftbark(i,framelength/2,48000))+1)
        if ((bit_alloc(i) ~= 0) & (bit_alloc(i) ~= 1))
            fwrite(fid, Data(indices(1):indices(end)) ,qbits);
        end
    end
end % end of frame loop

fclose(fid);

% RUN DECODER
disp('Decoding...');
p_decode(coded_filename,decoded_filename);

disp('Okay, all done!');





%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%          FFTBARK          %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function b=fftbark(bin,N,Fs)
% b=fftbark(bin,N,Fs)
% Converts fft bin number to bark scale
% N is the fft length
% Fs is the sampling frequency
f = bin*(Fs/2)/N;
b = 13*atan(0.76*f/1000) + 3.5*atan((f/7500).^2); 


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%          ENFRAME          %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function f=enframe(x,win,inc)
%ENFRAME split signal up into (overlapping) frames: one per row. F=(X,WIN,INC)
%
%	F = ENFRAME(X,LEN) splits the vector X up into
%	frames. Each frame is of length LEN and occupies
%	one row of the output matrix. The last few frames of X
%	will be ignored if its length is not divisible by LEN.
%	It is an error if X is shorter than LEN.
%
%	F = ENFRAME(X,LEN,INC) has frames beginning at increments of INC
%	The centre of frame I is X((I-1)*INC+(LEN+1)/2) for I=1,2,...
%	The number of frames is fix((length(X)-LEN+INC)/INC)
%
%	F = ENFRAME(X,WINDOW) or ENFRAME(X,WINDOW,INC) multiplies
%	each frame by WINDOW(:)

%	Copyright (C) Mike Brookes 1997
%
%      Last modified Tue May 12 13:42:01 1998
%
%   VOICEBOX home page: http://www.ee.ic.ac.uk/hp/staff/dmb/voicebox/voicebox.html
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   This program is free software; you can redistribute it and/or modify
%   it under the terms of the GNU General Public License as published by
%   the Free Software Foundation; either version 2 of the License, or
%   (at your option) any later version.
%
%   This program is distributed in the hope that it will be useful,
%   but WITHOUT ANY WARRANTY; without even the implied warranty of
%   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%   GNU General Public License for more details.
%
%   You can obtain a copy of the GNU General Public License from
%   ftp://prep.ai.mit.edu/pub/gnu/COPYING-2.0 or by writing to
%   Free Software Foundation, Inc.,675 Mass Ave, Cambridge, MA 02139, USA.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

nx=length(x);
nwin=length(win);
if (nwin == 1)
   len = win;
else
   len = nwin;
end
if (nargin < 3)
   inc = len;
end
nf = fix((nx-len+inc)/inc);
f=zeros(nf,len);
indf= inc*(0:(nf-1)).';
inds = (1:len);
f(:) = x(indf(:,ones(1,len))+inds(ones(nf,1),:));
if (nwin > 1)
    w = win(:)';
    f = f .* w(ones(nf,1),:);
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%          SCHROEDER        %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function m=Schroeder(freq,spl,downshift)
% Calculate the Schroeder masking spectrum for a given frequency and SPL

N = 2048;
f_kHz = [1:48000/N:48000/2];
f_kHz = f_kHz/1000;
A = 3.64*(f_kHz).^(-0.8) - 6.5*exp(-0.6*(f_kHz - 3.3).^2) + (10^(-3))*(f_kHz).^4;
f_Hz = f_kHz*1000;

% Schroeder Spreading Function
dz = bark(freq)-bark(f_Hz);
mask = 15.81 + 7.5*(dz+0.474) - 17.5*sqrt(1 + (dz+0.474).^2);

New_mask = (mask + spl - downshift);

m = New_mask;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%           BARK            %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function b=bark(f)
% b=bark(f)
% Converts frequency to bark scale
% Frequency should be specified in Hertz

b = 13*atan(0.76*f/1000) + 3.5*atan((f/7500).^2); 


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%        ALLOCATE           %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function x=allocate(y,b,N,Fs)
% x=allocate(y,b,N)
% Allocates b bits to the 25 subbands
% of y (a length N/2 MDCT, in dB SPL)

bits(floor(bark( (Fs/2)*[1:N/2]/(N/2) )) +1) = 0;

for i=1:N/2
bits(floor(bark( (Fs/2)*i/(N/2) )) +1) = max(bits(floor(bark( (Fs/2)*i/(N/2) )) +1) , ceil( y(i)/6 ));
end

indices = find(bits(1:end) < 2);
bits(indices(1:end)) = 0;

% NEED TO CALCULATE SAMPLES PER SUBBAND
n = 0:N/2-1;
f_Hz = n*Fs/N;
f_kHz = f_Hz / 1000;
A_f = 3.64*f_kHz.^-.8 - 6.5*exp(-.6*(f_kHz-3.3).^2) + 1e-3*f_kHz.^4;    % *** Threshold in Quiet
z = 13*atan(0.76*f_kHz) + 3.5*atan((f_kHz/7.5).^2);  % *** bark frequency scale
crit_band = floor(z)+1;
num_crit_bands = max(crit_band);
num_crit_band_samples = zeros(num_crit_bands,1);
for i=1:N/2
    num_crit_band_samples(crit_band(i)) = num_crit_band_samples(crit_band(i)) + 1;
end

x=zeros(1,25);
bitsleft=b;
[blah,i]=max(bits);
while bitsleft > num_crit_band_samples(i)
    [blah,i]=max(bits);
    x(i) = x(i) + 1;
    bits(i) = bits(i) - 1;
    bitsleft=bitsleft-num_crit_band_samples(i);
end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%        P_ENCODE           %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [Quantized_Gain,quantized_words]=p_encode(x2,Fs,framelength,bit_alloc,scalebits)

for i=1:floor(fftbark(framelength/2,framelength/2,Fs))+1
    indices = find((floor(fftbark([1:framelength/2],framelength/2,Fs))+1)==i);
    Gain(i) = 2^(ceil(log2((max(abs(x2(indices(1):indices(end))+1e-10))))));
    if Gain(i) < 1
        Gain(i) = 1;
    end
    x2(indices(1):indices(end)) = x2(indices(1):indices(end)) / (Gain(i)+1e-10);
    Quantized_Gain(i) = log2(Gain(i));
end
    
for i=1:length(x2)
        quantized_words(i) = midtread_quantizer(x2(i), max(bit_alloc(floor(fftbark(i,framelength/2,Fs))+1),0)+1e-10); % 03/20/03
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%    MIDTREAD_QUANTIZER     %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [ret_value] = midtread_quantizer(x,R)

Q = 2 / (2^R - 1);      
q = quant(x,Q);
s = q<0;    
ret_value = uint16(abs(q)./Q + s*2^(R-1));


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   MIDTREAD_DEQUANTIZER    %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [ret_value] = midtread_dequantizer(x,R)

sign = (2 * (x < 2^(R-1))) - 1;
Q = 2 / (2^R - 1); 

x_uint = uint32(x);
x = bitset(x_uint,R,0);      
x = double(x);

ret_value = sign * Q .* x;



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%         P_DECODE          %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function Fs=p_decode(coded_filename,decoded_filename)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%      READ FILE HEADER     %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fid = fopen(coded_filename,'r');
Fs          = fread(fid,1,'ubit16'); % Sampling Frequency
framelength = fread(fid,1,'ubit12'); % Frame Length
bitrate     = fread(fid,1,'ubit18'); % Bit Rate
scalebits   = fread(fid,1,'ubit4' ); % Number of Scale Bits per Sub-Band
num_frames  = fread(fid,1,'ubit26'); % Number of frames


for frame_count=1:num_frames
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %     READ FILE CONTENTS    %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    qbits = sprintf('ubit%i', scalebits);
    gain = fread(fid,25,qbits);
    bit_alloc = fread(fid,25,'ubit4');
    for i=1:floor(fftbark(framelength/2,framelength/2,Fs))+1
        indices = find((floor(fftbark([1:framelength/2],framelength/2,Fs))+1)==i);
        if ((bit_alloc(i) ~= 0) & (bit_alloc(i) ~= 1))
            qbits = sprintf('ubit%i', bit_alloc(i)); 
            InputValues(indices(1):indices(end)) = fread(fid, length(indices) ,qbits);
        else
            InputValues(indices(1):indices(end)) = 0;
        end
    end



    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %     DEQUANTIZE VALUES     %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    for i=1:length(InputValues)
        if InputValues(i) ~= 0
            if max(bit_alloc(floor(fftbark(i,framelength/2,Fs))+1),0) ~= 0
                InputValues(i) = midtread_dequantizer(InputValues(i),...
                    max(bit_alloc(floor(fftbark(i,framelength/2,Fs))+1),0));
            end
        end
    end

    for i=1:25
        gain2(i) = 2^gain(i);
    end


    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %         APPLY GAIN        %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    for i=1:floor(fftbark(framelength/2,framelength/2,Fs))+1
        indices = find((floor(fftbark([1:framelength/2],framelength/2,Fs))+1)==i);
        InputValues(indices(1):indices(end)) = InputValues(indices(1):indices(end)) * gain2(i);
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %       INVERSE MDCT        %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    x2((frame_count-1)*framelength+1:frame_count*framelength) = imdct(InputValues(1:framelength/2));
end

status = fclose(fid);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%     RECOMBINE FRAMES      %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
x3 = zeros(1,(length(x2)-1)/2+1);
for i=0:0.5:floor(length(x2)/(2*framelength))-1
    x3(i*framelength+1 : (i+1)*framelength) = x3(i*framelength+1 : (i+1)*framelength) + x2((2*i)*framelength+1 : (2*i+1)*framelength);
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%        WRITE FILE         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
wavwrite(x3/2,Fs,decoded_filename);



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%           MDCT            %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function y = mdct(x)

x=x(:);
N=length(x);
n0 = (N/2+1)/2;
wa = sin(([0:N-1]'+0.5)/N*pi);
y = zeros(N/2,1);

x = x .* exp(-j*2*pi*[0:N-1]'/2/N) .* wa;

X = fft(x);

y = real(X(1:N/2) .* exp(-j*2*pi*n0*([0:N/2-1]'+0.5)/N));
y=y(:);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%          IMDCT            %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function y = imdct(X)

X=X(:);
N = 2*length(X);
ws = sin(([0:N-1]'+0.5)/N*pi);
n0 = (N/2+1)/2;
Y = zeros(N,1);

Y(1:N/2) = X;
Y(N/2+1:N) = -1*flipud(X);
Y = Y .* exp(j*2*pi*[0:N-1]'*n0/N);
y = ifft(Y);
y = 2*ws .* real(y .* exp(j*2*pi*([0:N-1]'+n0)/2/N));




