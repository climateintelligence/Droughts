# source all (data) fig and table

library(fs)

# source("R/11-paper/data-01-trends.R")
# source("R/11-paper/data-02-eofpca-clust.R")
# source("R/11-paper/data-03-correlations.R")
# source("R/11-paper/data-04-for-spatial-consistency.R")


r_files <- dir_ls("R/11-paper/", regexp = "fig|table")

for(i_fn in r_files) source(i_fn)