# Date Time Stamp
dts() { date +%Y-%m-%d-%H-%M-%S; }

# Date mkdir
dmkdir() { mkdir $(dts); }

# Create a new experiments folder
cd experiments
name=$(dts);
mkdir $name
cd ..
pwd

# Generate results files
rm plots/results/*
python output_multiple_crop.py

# Generate error files based on results
rm plots/comparisons/errors_i/*.png
rm plots/comparisons/errors_o/*.png
python area_extractor.py
montage $(ls -1 plots/comparisons/errors_i/*.png | sort -g) -geometry +10+10 plots/comparisons/errors_i/input_errors.jpg
montage $(ls -1 plots/comparisons/errors_o/*.png | sort -g) -geometry +10+10 plots/comparisons/errors_o/output_errors.jpg

mv plots/comparisons/errors_i/input_errors.jpg experiments/$name
mv plots/comparisons/errors_o/output_errors.jpg experiments/$name

# Generate cams from errors
rm plots/cams/*
rm plots/heatmaps/*
rm plots/backprops/*
cd visualize
python visualize.py
cd ..

montage $(ls -1 plots/heatmaps/*.png | sort -g) -geometry +10+10 plots/heatmaps/heatmap.jpg
montage $(ls -1 plots/backprops/*.png | sort -g) -geometry +10+10 plots/backprops/backprop.jpg
#montage plots/heatmaps/*.png -geometry +10+10 plots/heatmaps/h_error.jpg

# Move cams to experiment folder
mv plots/heatmaps/heatmap.jpg experiments/$name
mv plots/backprops/backprop.jpg experiments/$name
