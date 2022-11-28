# based off of a helpful tutorial
# https://tomazweiss.github.io/blog/object_detection/

#1. download and library packages----
devtools::install_github("bnosac/image",
                         subdir = "image.darknet",
                         build_vignettes = TRUE)

library(image.darknet)
library(Rcpp)
library(dplyr)
library(tidyr)
library(here)

#part of this may rely on rtools which is not available on mac
#https://clanfear.github.io/CSSS508/docs/compiling.html this may get around this
#but we can do the work without the functions - just makes things easier

# 2. define the model ----
#Define Model - here it is Tiny Yolo
detect_model <- image_darknet_model(type = 'detect',
                                    model = "tiny-yolo-voc.cfg",
                                    weights = system.file(package="image.darknet", "models", "tiny-yolo-voc.weights"),
                                    labels = system.file(package="image.darknet", "include", "darknet", "data", "voc.names"))

#3. Set up folders ----
# folder for output images with predictions
dir.create('img')

# folder with input images
path <- here("img")

# get all pngs and jpgs
images <- dir(path = path, pattern = "\\.png|\\.jpg|\\.jpeg")

# folder for output images with predictions
dir.create('pred')

#4. Functions to apply method to all images in folder ----
# function to be applied to images
detect_objects <- function(x) {

  filename <- paste(path, x, sep = "/")

  prediction <- image_darknet_detect(
    file = filename,
    object = detect_model,
    threshold = 0.19
  )

  file.rename("predictions.png", paste0("pred/", x))
  return(prediction)
}

#functions to handle the code being stored correctly
cppFunction('void redir(){FILE* F=freopen("capture.txt","w+",stdout);}')
cppFunction('void resetredir(){FILE* F=freopen("CON","w+",stdout);}')

#5. Run the model ----
redir();
d <- lapply(images, detect_objects)
resetredir();

#6. reformat the text outputs----
# Read in the output file
d <- data.frame(txt = unlist(readLines("capture.txt")))

## Take out all the lines that we don't need.
d <- d %>%
  filter(!grepl("Boxes", txt)) %>%
  filter(!grepl("pandoc", txt)) %>%
  filter(!grepl("unnamed", txt))

## Find the lines that contain the file names. Make a logical vector called "isfile"
d$isfile <- grepl(path, d$txt)

## Take out the path and keep only the file names
d$txt <- gsub(paste0(path, '/'), "", d$txt)

## Make a column called file that contains either file names or NA
d$file <- ifelse(d$isfile, d$txt, NA)

## All the other lines of text refer to the objects detected
d$object <- ifelse(!d$isfile, d$txt, NA)

## Fill down
d <- tidyr::fill(d, "file")

## Take out NAs and select the last two columns
d <- na.omit(d)[, 3:4]

# Separate the text that is held in two parts
d <- d %>% separate(file, into = c("file", "time"), sep = ":")
d <- d %>% separate(object, into = c("object", "prob"), sep = ":")
d <- d %>% filter(!is.na(prob))

# Keep only the prediction time
d$time <- gsub("Predicted in (.+).$", "\\1", d$time)

# Convert probabilities to numbers
d$prob <- as.numeric(sub("%", "", d$prob)) / 100

# PART 2 IMAGE CLASSIFICATION ----
#Define model
model <-system.file(package="image.darknet","include","darknet","cfg","tiny.cfg")

weights <-system.file(package="image.darknet","models","tiny.weights")

labels <-system.file(package="image.darknet","include","darknet","data","imagenet.shortnames.list")

labels <-readLines(labels)

darknet_tiny <-image_darknet_model(type = 'classify',
                                   model = model,
                                   weights = weights,
                                   labels = labels)

## Classify new images alongside the model
# folder with input images
path <- here("img")

# get all pngs and jpgs
images <- dir(path = path, pattern = "\\.png|\\.jpg|\\.jpeg")

classified <- NULL

for(i in 1:length(images)){

  filename <- paste(path, images[i], sep = "/")

  tmp_df <- image_darknet_classify(file = filename,
                                   object = darknet_tiny)

  tmp_df <- data.frame(image = tmp_df$file,
                       tmp_df$type)

  classified <- dplyr::bind_rows(classified,
                                 tmp_df)

}


