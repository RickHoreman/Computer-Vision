# Getting Started

First things first I would recommend you have a look at the documentation folder, which currently contains a worklog.txt I kept while working on everything, and a proper report.

Then if you want to see the thing in action run the following files in this order:
1. ImageProcessing.py - This will make 4 .npy files in the data folder
2. trainCLF.py - This will make a clf.joblib file in the data folder (with the current version these files add up to be <1GB but do mind you need _some_ free space)
3. testCLF.py - This will tell you the accuracy in % and exactly which images it failed to identify, what it thought they were and what they actually are.

There are some values in these files that can be tweaked to change the used methods, resolution, etc. I'll leave these at the ones that gave me the best results.
If I had the time I would have loved to make a proper testing interface like a single function call with some changeable parameters, or maybe even running the .py file with certain parameters, but the deadline is (as of writing) tomorrow, sooooo...





//// Under here is the "old" README containing the first assignment of choosing what you were going to do (sorry for the language inconsistency) and the planning.

# Computer-Vision

Voor de computer vision eindopdracht ga ik voor keuze 2: een challenge.

Hiervoor ga ik de volgende challenge uitvoeren: https://www.kaggle.com/c/street-view-getting-started-with-julia/overview

Met de bijbehorende dataset, te vinden op http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/

En de gerelateerde paper, Character Recognition In Natural Images: https://www.scitepress.org/Papers/2009/17701/17701.pdf

De paper maakt gebruik van de volgende methoden:
- Shape Contexts
- Geometric Blur
- Scale Invariant Feature Transform
- Spin image
- Maximum Response of filters
- Patch descriptor

Dit zijn er alleen een beetje veel voor deze opdracht dus ik ga hier waarschijnlijk een stuk of 2/3 van moeten uitkiezen om de opdracht mee te doen. Deze keuze moet ik nog maken.


# Planning

Must have:
- Character recognition through bag-of-visual-words using (potentially multiple) feature detection method(s).
- Some better than random guess accuracy.

Should have:
- A usable character recognision on the fnt dataset

Could have:
- One or more of the actual methods for the paper (these turned out to be more complicated than I thought)

Wont Have:
- Any similarities between mine and Joris Heemskerk's work (same challenge). :)

Week 4:
- Work on weekly assignments (these are helpful for knowing what to do for this challenge)

Week 5:
- Finish weekly assignments

Week 6:
- Monday/Tuesday: Actually finish last weekly assignment, and make all of the image loading and preprocessing stuff
- Saturday/Sunday: Looking into the methods from the paper (took way too long)

Weeks 7:
- Monday: More looking into the methods from the paper, and ultimately deciding to start working on a simpler version with the stuff I learned from the weekly assignments
- Tuesday: Start work on simple version
- Wednesday: R2D2 stuffs
- Thursday: Finish initial simple version
- Friday: Debug absurdly high initial accuracy
- Saturday/Sunday: Improve on simple version
