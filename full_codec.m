function decodedFile = full_codec(originalFile,bitrate,decodedFile,codedFile)
% FULL_CODEC encodes and decodes an audio file
%   decodedFile = FULL_CODEC(originalFile) encodes and decodes original file
%   decodedFile = FULL_CODEC(originalFile,bitrate) lets you specify the
%   encoded bitrate in bits per second (128000 default)
%   FULL_CODEC(originalFile,bitrate,decodedFile) lets you specify the
%   decoded file
%   FULL_CODEC(originalFile,bitrate,decodedFile,codedFile) also lets you
%   specify the encoded file
%
%   This code is based loosely on the assignments in the textbook:
%   Bosi, Marina, and Richard E. Goldberg. "Introduction to digital audio
%     coding and standards". Springer, 2003.
% 
%   See also AUDIOWRITE

% Github location:
%   https://github.com/JonBoley/SimpleCodec.git
% Bug reports:
%   https://github.com/JonBoley/SimpleCodec/issues
% 
% Copyright 2002, Jon Boley (jdb@jboley.com)

if nargin<4
    codedFile = 'yourfile.jon';
end
if nargin<3
    decodedFile = 'yourfile_decoded.wav';
end
if nargin<2
    bitrate = 128000;
end
if nargin<1
    originalFile = 'yourfile.wav';
end

scalebits = 4;
N = 2048; % framelength

[Y,Fs] = audioread(originalFile);
Y = Y(:,1); % just use the first channel

sig=sin(2*pi*1000*(1:N/2)/Fs);
win=(0.5 - 0.5*cos((2*pi*((1:(N/2))-0.5))/(N/2))); 
fftmax = max(abs(fft(sig.*win))); % defined as 96dB

%   Enframe Audio   
frames = enframe(Y,N,N/2);

% Write File Header 
fid = fopen(codedFile,'w');
fwrite(fid, Fs, 'ubit16'); % Sampling Frequency
fwrite(fid, N, 'ubit12'); % Frame Length
fwrite(fid, bitrate, 'ubit18'); % Bit Rate
fwrite(fid, scalebits, 'ubit4'); % Number of Scale Bits per Sub-Band
fwrite(fid, length(frames(:,1)), 'ubit26'); % Number of frames
    
numBands = floor(fftbark(N/2,N/2,Fs))+1;

%   Computations    
for frame_count=1:length(frames(:,1))

    if mod(frame_count,10) == 0
        outstring = sprintf('Now Encoding Frame %i of %i', frame_count, length(frames(:,1)));
        disp(outstring);
    end
    
    fft_frame = fft(frames(frame_count,:));

    if fft_frame == zeros(1,N)
        Gain = zeros(1,numBands);
        bit_alloc = zeros(1,numBands);
    else    
        len = length(fft_frame);
        peak_width = zeros(1,len);

        % Find Peaks
        centers = find(diff(sign(diff( abs(fft_frame).^2) )) == -2) + 1;
        spectral_density = zeros(1,length(centers));
    
        peak_max = NaN*ones(size(centers));
        peak_min = NaN*ones(size(centers));
        for k=1:numel(centers)
            peak_max(k) = centers(k) +2;
            peak_min(k) = centers(k) - 2;
            peak_width(k) = peak_max(k) - peak_min(k);
            
            for j=peak_min(k):peak_max(k)
                if (j > 0) && (j < N)
                    spectral_density(k) = spectral_density(k) + abs(fft_frame(j))^2;
                end
            end
        end
            
        % This gives the squared amplitude of the original signal
        % (this is here just for educational purposes)
        modified_SD = spectral_density / ((N^2)/8);
        SPL = 96 + 10*log10(modified_SD);
        
        % TRANSFORM FFT'S TO SPL VALUES
        fft_spl = 96 + 20*log10(abs(fft_frame)/fftmax);
    
    
        % Threshold in Quiet
        f_kHz = (1:Fs/N:Fs/2)/1000;
        A = 3.64*(f_kHz).^(-0.8) - 6.5*exp(-0.6*(f_kHz - 3.3).^2) + (10^(-3))*(f_kHz).^4;
    
        % Masking Spectrum
        big_mask = max(A,Schroeder(Fs,N,centers(1)*Fs/N,fft_spl(centers(1)),...
            14.5+bark(centers(1)*Fs/N)));
        for peak_count=2:sum(centers*Fs/N<=Fs/2)
            big_mask = max(big_mask,Schroeder(Fs,N,centers(peak_count)*Fs/N,...
                fft_spl(centers(peak_count)), 14.5+bark(centers(peak_count)*Fs/N)));
        end

        % Signal Spectrum - Masking Spectrum (with max of 0dB)
        New_FFT = fft_spl(1:N/2)-big_mask;
        New_FFT_indices = find(New_FFT > 0);
        New_FFT2 = zeros(1,N/2);
        for ii=1:length(New_FFT_indices)
            New_FFT2(New_FFT_indices(ii)) = New_FFT(New_FFT_indices(ii));
        end
    
        if frame_count == 55
            semilogx(0:(Fs/2)/(N/2):Fs/2-1,fft_spl(1:N/2),'b');
            hold on;
            semilogx(0:(Fs/2)/(N/2):Fs/2-1,big_mask,'m');
            hold off;
            title('Signal (blue) and Masking Spectrum (pink)');
            figure;
            semilogx(0:(Fs/2)/(N/2):Fs/2-1,New_FFT2);
            title('SMR');
            figure;
            stem(allocate(New_FFT2,bitrate,scalebits,N,Fs));
            title('Bits perceptually allocated');
        end
        
        bit_alloc = allocate(New_FFT2,bitrate,scalebits,N,Fs);
    
        [Gain,Data] = p_encode(mdct(frames(frame_count,:)),Fs,N,bit_alloc,scalebits);
    end % end of If-Else Statement        
        
    % Write Audio Data to File
    qbits = sprintf('ubit%i', scalebits);
    fwrite(fid, Gain, qbits);
    fwrite(fid, bit_alloc, 'ubit4');
    for ii=1:numBands
        indices = find((floor(fftbark(1:N/2,N/2,Fs))+1)==ii);
        qbits = sprintf('ubit%i', bit_alloc(ii)); % bits(floor(fftbark(i,framelength/2,48000))+1)
        if bit_alloc(ii)>0
            fwrite(fid, Data(indices(1):indices(end)) ,qbits);
        end
    end
end % end of frame loop

fclose(fid);

% RUN DECODER
disp('Decoding...');
p_decode(codedFile,decodedFile);

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
function New_mask = Schroeder(Fs,N,freq,spl,downshift)
% Calculate the Schroeder masking spectrum for a given frequency and SPL

f_Hz = 1:Fs/N:Fs/2;

% Schroeder Spreading Function
dz = bark(freq)-bark(f_Hz);
mask = 15.81 + 7.5*(dz+0.474) - 17.5*sqrt(1 + (dz+0.474).^2);

New_mask = (mask + spl - downshift);


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
function x=allocate(y,bitrate,scalebits,N,Fs)
% x=allocate(y,b,sb,N,Fs)
% Allocates b bits to the 25 subbands
% of y (a length N/2 MDCT, in dB SPL)

% artifact reduction (reduce high frequency bands at low bitrates)
numBandsToIgnore = 2*floor(128000/bitrate);

num_subbands = floor(fftbark(N/2,N/2,Fs))+1;
bits_per_frame = floor(((bitrate/Fs)*(N/2)) - (scalebits*num_subbands));
        
bits(floor(bark( (Fs/2)*(1:N/2)/(N/2) )) +1) = 0;

for ii=1:N/2
    bits(floor(bark( (Fs/2)*ii/(N/2) )) +1) = max(bits(floor(bark( (Fs/2)*ii/(N/2) )) +1) , ceil( y(ii)/6 ));
end

indices = find(bits(1:end) < 2);
bits(indices(1:end)) = 0;
bits(end-numBandsToIgnore:end) = 0; % artifact reduction

% NEED TO CALCULATE SAMPLES PER SUBBAND
n = 0:N/2-1;
f_Hz = n*Fs/N;
f_kHz = f_Hz / 1000;
z = 13*atan(0.76*f_kHz) + 3.5*atan((f_kHz/7.5).^2);  % *** bark frequency scale
crit_band = floor(z)+1;
num_crit_bands = max(crit_band);
num_crit_band_samples = zeros(num_crit_bands,1);
for ii=1:N/2
    num_crit_band_samples(crit_band(ii)) = num_crit_band_samples(crit_band(ii)) + 1;
end

x = zeros(1,num_subbands);
bitsleft = bits_per_frame;
[~,ii]=max(bits);
while bitsleft > num_crit_band_samples(ii)
    [~,ii]=max(bits);
    x(ii) = x(ii) + 1;
    bits(ii) = bits(ii) - 1;
    bitsleft=bitsleft-num_crit_band_samples(ii);
end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%        P_ENCODE           %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [Quantized_Gain,quantized_words]=p_encode(x2,Fs,framelength,bit_alloc,~)

N = floor(fftbark(framelength/2,framelength/2,Fs))+1;
Gain = ones(N,1);
Quantized_Gain = zeros(N,1);

for ii=1:N
    indices = find((floor(fftbark(1:framelength/2,framelength/2,Fs))+1)==ii);
    Gain(ii) = 2^(ceil(log2((max(abs(x2(indices(1):indices(end))+1e-10))))));
    if Gain(ii) < 1
        Gain(ii) = 1;
    end
    x2(indices(1):indices(end)) = x2(indices(1):indices(end)) / (Gain(ii)+1e-10);
    Quantized_Gain(ii) = log2(Gain(ii));
end
    
quantized_words = zeros(size(x2));
for ii=1:numel(x2)
    quantized_words(ii) = midtread_quantizer(x2(ii), max(bit_alloc(floor(fftbark(ii,framelength/2,Fs))+1),0)+1e-10); % 03/20/03
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

% birdie reduction constant (may add echo if too high)
rampConstant = 0.5; % [0.0 - 1.0]

% Read file header
fid = fopen(coded_filename,'r');
Fs          = fread(fid,1,'ubit16'); % Sampling Frequency
framelength = fread(fid,1,'ubit12'); % Frame Length
bitrate     = fread(fid,1,'ubit18'); % Bit Rate
scalebits   = fread(fid,1,'ubit4' ); % Number of Scale Bits per Sub-Band
num_frames  = fread(fid,1,'ubit26'); % Number of frames

prevInputValues = zeros(1,framelength/2);
for frame_count=1:num_frames
    
    % Read file contents
    qbits = sprintf('ubit%i', scalebits);
    gain = fread(fid,25,qbits);
    bit_alloc = fread(fid,25,'ubit4');
    for ii=1:floor(fftbark(framelength/2,framelength/2,Fs))+1
        indices = find((floor(fftbark(1:framelength/2,framelength/2,Fs))+1)==ii);
        if bit_alloc(ii) > 0
            qbits = sprintf('ubit%i', bit_alloc(ii)); 
            InputValues(indices(1):indices(end)) = fread(fid, length(indices) ,qbits);
        else
            InputValues(indices(1):indices(end)) = 0;
        end
    end

    % Dequantize values
    for ii=1:length(InputValues)
        if InputValues(ii) ~= 0
            if max(bit_alloc(floor(fftbark(ii,framelength/2,Fs))+1),0) ~= 0
                InputValues(ii) = midtread_dequantizer(InputValues(ii),...
                    max(bit_alloc(floor(fftbark(ii,framelength/2,Fs))+1),0));
            end
        end
    end

    gain2 = zeros(size(gain));
    for ii=1:25
        gain2(ii) = 2^gain(ii);
    end

    % Apply gain
    for ii=1:floor(fftbark(framelength/2,framelength/2,Fs))+1
        indices = find((floor(fftbark(1:framelength/2,framelength/2,Fs))+1)==ii);
        InputValues(indices(1):indices(end)) = InputValues(indices(1):indices(end)) * gain2(ii);
     end

    % Apply birdie reduction
    for ii=1:floor(fftbark(framelength/2,framelength/2,Fs))+1    
        if bit_alloc(ii)<1
            InputValues(indices(1):indices(end)) = prevInputValues(indices(1):indices(end)) * rampConstant;
        end
    end
    
    % save this frame
    prevInputValues = InputValues;
    
    % Inverse MDCT
    x2((frame_count-1)*framelength+1:frame_count*framelength) = imdct(InputValues(1:framelength/2));
end

fclose(fid);
% Recombine frames
x3 = zeros(1,ceil((length(x2)-1)/2+1));
for ii=0:0.5:floor(length(x2)/(2*framelength))-1
    x3(ii*framelength+1 : (ii+1)*framelength) = x3(ii*framelength+1 : (ii+1)*framelength) + x2((2*ii)*framelength+1 : (2*ii+1)*framelength);
end

% Write file
audiowrite(decoded_filename,x3,Fs);



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%           MDCT            %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function y = mdct(x)

x=x(:);
N=length(x);
n0 = (N/2+1)/2;
wa = sin(((0:N-1)'+0.5)/N*pi);

x = x .* exp(-1i*2*pi*(0:N-1)'/2/N) .* wa;

X = fft(x);

y = real(X(1:N/2) .* exp(-1i*2*pi*n0*((0:N/2-1)'+0.5)/N));
y=y(:);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%          IMDCT            %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function y = imdct(X)

X=X(:);
N = 2*length(X);
ws = sin(((0:N-1)'+0.5)/N*pi);
n0 = (N/2+1)/2;
Y = zeros(N,1);

Y(1:N/2) = X;
Y(N/2+1:N) = -1*flipud(X);
Y = Y .* exp(1i*2*pi*(0:N-1)'*n0/N);
y = ifft(Y);
y = 2*ws .* real(y .* exp(1i*2*pi*((0:N-1)'+n0)/2/N));
