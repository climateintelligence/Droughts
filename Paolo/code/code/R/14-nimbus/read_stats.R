# function to read in stats csv file

# NO REGION!

library(data.table)
library(lubridate)
library(magrittr)
library(forcats)
library(mitmatmisc)
library(raster)

read_stats <- function(fn_stats, fn_sample_raster){
  
  dat <- fread(fn_stats)
  
  dat2 <- dat[, .(ff = basename(fn_stats) %>% fs::path_ext_remove(),
          date = as.Date(ymd_hms(file)),
          value, 
          count)]
  
  # summarize nodata/water into 1 class
  ncell <- ncell(raster(fn_sample_raster))
  dat2[, count_frac := count / ncell]
  dat2[, value_fct := fct_collapse(as.character(value),
                                      nodata_water = c("0", "4", "5"),
                                      snow = "1",
                                      land = "2",
                                      clouds = "3")]

  dat2_class <- dat2[, 
                     .(count_frac = sum(count_frac),
                       count= sum(count)),
                     .(ff, date, value_fct)]
  
  # fill in 0's for missing dates 
  CJ(ff = unique(dat2_class$ff),
     value_fct = unique(dat2_class$value_fct),
     date = seq(min(dat2_class$date), max(dat2_class$date), by = "day")) %>% 
    merge(dat2_class, all.x = T) -> dat2_class_full
  # with(dat_stats_class_full, table(ff, value_f, is.na(count_frac)))
  
  dat2_class_full[is.na(count_frac), count_frac := 0]
  dat2_class_full[is.na(count), count := 0]
  
  
  
  # plot helpers --------------------------------------------------------------------
  
  add_hydro_year(dat2_class_full)
  add_month_fct(dat2_class_full, 10)
  
  
  cellsize_km2 <- prod(res(raster(fn_sample_raster))/1000) # km2
  dat2_class_full[, area_km2 := count * cellsize_km2]
  
  return(dat2_class_full)
  
}