# export QC information for MeteoSwiss

library(data.table)
library(magrittr)
library(fs)
library(readxl)
library(writexl)



dat_meta_1 <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/02_JOIN_REGIONS/1961-2020/r-data/meta_long_HN_HS.rds")
dat_meta_2 <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/02_JOIN_REGIONS/1787hn_1879hs-1960/r-data/meta_long_HN_HS.rds")
dat_meta_all <- unique(rbind(dat_meta_1, dat_meta_2))
setkey(dat_meta_all, Provider, Name)

stns_mswiss <- dat_meta_all[Provider == "CH_METEOSWISS", Name]

# temporal consistency ----------------------------------------------------

dat_tc <- read_excel("manual-qc/v02/temporal-consistency_prefilled-v01_filled.xlsx")
setDT(dat_tc)
dat_tc %>% str
dat_tc[, Date_error := as.Date(Date_error)]
dat_tc[, Date := as.Date(Date)]
setkey(dat_tc, Name, Date)

dat_tc[Name %in% stns_mswiss & (HS_error == 1 | HN_error == 1),
       .(idn)] %>% unique %>% 
  merge(dat_tc, by = "idn") -> dat_tc_out

write_xlsx(dat_tc_out, 
           "/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/09_EXPORT/aux-info-meteoswiss/temporal-consistency.xlsx")


# zero-NA -----------------------------------------------------------------

dat_0na_overview <- read_excel("manual-qc/v02/zero-NA-overview_prefilled-v01_filled.xlsx")
setDT(dat_0na_overview)
files_sub <- fs::path("manual-qc/v02/zero-NA_prefilled-v01_filled/", 
                      dat_0na_overview[zero_NA == 1, Name], ext = "xlsx")
l <- lapply(files_sub, read_excel)
names(l) <- dat_0na_overview[zero_NA == 1, Name]
dat_0na <- rbindlist(l, fill = T, use.names = T, idcol = "Name")
dat_0na[, Date := as.Date(Date)]
setkey(dat_0na, Name, Date)

dat_0na_out <- dat_0na[Name %in% stns_mswiss & zeroNA == 1 & HS == 0]
dat_0na[remove_too == 1 & Name %in% stns_mswiss]

dat_0na_overview_out <- dat_0na_overview[Name %in% stns_mswiss & zero_NA == 1]

write_xlsx(dat_0na_overview_out, 
           "/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/09_EXPORT/aux-info-meteoswiss/zeroNA-overview.xlsx")


write_xlsx(dat_0na_out, 
           "/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/09_EXPORT/aux-info-meteoswiss/zeroNA-station-series.xlsx")

for(i_stn in unique(dat_0na_overview_out$Name)){
  
  file_copy(
    path("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/03_QC1/v02/fig/zero-NA/", i_stn, ext = "pdf"),
    path("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/09_EXPORT/aux-info-meteoswiss/zeroNA-fig/", i_stn, ext = "pdf")
  )
  
}




# suspicious and to check series ------------------------------------------
# 
# dat_sac_overview <- read_excel("manual-qc/v02/series-to-check-overview_prefilled-v01_filled.xlsx")
# setDT(dat_sac_overview)
# 
# dat_sac_overview[Name %in% stns_mswiss]
# 
# files_sub <- fs::path("manual-qc/v02/series-to-check_prefilled-v01_filled/", 
#                       dat_sac_overview[remove_some_values == 1, Name], ext = "xlsx")
# l <- lapply(files_sub, read_excel)
# names(l) <- dat_sac_overview[remove_some_values == 1, Name]
# dat_sac <- rbindlist(l, fill = T, use.names = T, idcol = "Name")
# dat_sac[, Date := as.Date(Date)]
# setkey(dat_sac, Name, Date)
# 
# dat_sac[Name %in% stns_mswiss & to_remove == 1]
# 
# 
# # remove whole series and suspicious ones
# stns_to_remove <- unique(dat_sac_overview[remove_whole_series == 1 | suspicious == 1, Name])
# dat_all <- dat_all[!.(stns_to_remove)]
# 

