# source all (data) fig and table

library(fs)

# source("R/11-paper/sensitivity-gapfill/sg_data-01-trends.R")
# source("R/11-paper/sensitivity-gapfill/sg_data-02-eofpca-clust.R")
# source("R/11-paper/sensitivity-gapfill/sg_data-03-correlations.R")



r_files <- dir_ls("R/11-paper/sensitivity-gapfill/", regexp = "sg_fig|sg_table")

for(i_fn in r_files) {
  print(i_fn)
  source(i_fn)
}
