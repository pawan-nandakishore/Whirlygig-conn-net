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
montage plots/comparisons/errors_i/*.png -geometry +10+10 plots/comparisons/errors_i/input_errors.jpg
montage plots/comparisons/errors_o/*.png -geometry +10+10 plots/comparisons/errors_o/output_errors.jpg

mv plots/comparisons/errors_i/input_errors.jpg experiments/$name
mv plots/comparisons/errors_o/output_errors.jpg experiments/$name

# Generate cams from errors
cd visualize
python visualize.py
cd ..

montage plots/heatmaps/*.png -geometry +10+10 plots/heatmaps/heatmap.jpg
montage plots/backprops/*.png -geometry +10+10 plots/backprops/backprop.jpg
#montage plots/heatmaps/*.png -geometry +10+10 plots/heatmaps/h_error.jpg

# Move cams to experiment folder
mv plots/heatmaps/heatmap.jpg experiments/$name
mv plots/backprops/backprop.jpg experiments/$name
