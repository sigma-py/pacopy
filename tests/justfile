prepare:
	for file in fig*.png; do \
		convert -resize 500 $$file $$file; \
	done
	for file in fig*.png; do \
	  optipng -quiet $$file; \
	done

apng: prepare
	# /kc is essential for the frames not to overlap
	apngasm out.png fig*.png /kc /kp

gif: prepare
	# <https://stackoverflow.com/a/23285387/353337>
	convert -dispose 2 -delay 10 -loop 0 fig*.png out.gif

webp: prepare
	img2webp fig*.png -min_size -lossy -o out.webp
