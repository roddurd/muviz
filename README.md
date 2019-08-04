# slapper
My initial vision was to have an app which can implement and automate popular hip hop music video visual effects (glitches, trippy coloring, snap zooms on the beat, etc.) and speed up workflow, but I am currently <i>very</i> far from that goal.

For now, it's just a CLI for testing my image/video manipulation libraries. 


# Features
- <b>DeepDream for video</b>: pretty much just a clone of Hvass Lab's DeepDream code, connected to two video <-> frame images conversion libraries I wrote.
![](deepdream gif.gif)
- <b>Horizontal glitch effect</b>: Ya the title pretty much says it. I just shift random rows of pixels by 50.
![](hor glitch gif.gif)
- <b>Faded effect</b>: Kind of like seeing double, but its like quadruple and the duplicate images are just slightly transparent and incrementally zoomed in versions of the frame.

- Ya that's kind of it rn. I had another function that draws random fractures on the screen but its kinda boring. And some very basic img manipulation functions like zooming from center, opacity, brighten/darken, grey/R/G/B-scale, etc.

