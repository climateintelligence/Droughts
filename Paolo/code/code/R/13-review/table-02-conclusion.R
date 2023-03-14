

library(data.table)
library(magrittr)
library(lubridate)
library(ggplot2)
library(forcats)
library(foreach)
library(patchwork)
library(flextable)
library(officer)

dat_meta_clust <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/PAPER/02_review/rds/meta-with-cluster-01.rds")
dat_month <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/PAPER/02_review/rds/data-01-monthly.rds")
dat_seasonal <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/PAPER/02_review/rds/data-02-seasonal.rds")

load("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/PAPER/02_review/rds/trends-01-1971-2019-ols-gls.rda")



# for conclusion

dat_table <- rbind(
  dat_seasonal_gls[!startsWith(variable, "SCD") & term == "year0",
                   .(Name, variable, estimate)],
  dat_seasonal_ols[startsWith(variable, "SCD") & term == "year0",
                   .(Name, variable, estimate)]
) %>% merge(dat_meta_clust, by = "Name")


dat_table[, ns_fct := fct_collapse(cluster_fct,
                                   "North" = c("NE", "NW", "North & high Alpine"),
                                   "South" = c("South & high Alpine", "SE"))]
dat_table[, elev_fct := cut(Elevation, 0:3*1000, dig.lab = 6)]

dat_conclusion_table <- dat_table[,
                                  .(nn = .N,
                                    mean = 10*mean(estimate), 
                                    min = 10*min(estimate), 
                                    max = 10*max(estimate),
                                    sd = 10*sd(estimate)),
                                  .(elev_fct, ns_fct, variable)]

dat_conclusion_table[, mmm := sprintf("%0.1f (%0.1f, %0.1f)", mean, min, max)]


dat_out <- dat_conclusion_table %>% 
  dcast(elev_fct + ns_fct ~ variable, value.var = c("mmm", "nn"))
nn_stn <- dat_out[, 
                  paste0(apply(.SD, 1, min, na.rm = T), "-",apply(.SD, 1, max, na.rm = T)), 
                  .SDcols = grep("nn_", names(dat_out), value = T)]
dat_out[, range_nn := nn_stn]

dat_header <- data.table(col_keys = names(dat_out))
dat_header <- dat_header[!startsWith(col_keys, "nn_")]
dat_header[, row1 := gsub("mmm_", "", col_keys)]
dat_header[, c("row1", "row2") := tstrsplit(row1, "_")]
dat_header[1:2, row1 := c("Elevation [m]", "Region")]
dat_header[1:2, row2 := c("", "")]
dat_header[.N, row1 := "# series"]
dat_header[.N, row2 := "(range)"]

dat_out %>% 
  flextable(col_keys = c("elev_fct", "ns_fct", "range_nn", 
                         "mmm_meanHS_DJF", "mmm_meanHS_MAM", "mmm_meanHS_NDJFMAM", "mmm_maxHS_NDJFMAM",
                         "mmm_SCD_NDJF", "mmm_SCD_MAM", "mmm_SCD_NDJFMAM")) %>% 
  set_header_df(dat_header) %>% 
  theme_booktabs() %>% 
  merge_v(j = "elev_fct") %>% 
  valign(j = "elev_fct", valign = "top") %>% 
  align(align = "left", part = "all") %>% 
  font(fontname = "Times New Roman", part = "all") %>% 
  fontsize(size = 10, part = "all") %>% 
  fix_border_issues() %>% 
  autofit() -> ft

read_docx() %>% 
  body_add_flextable(ft) %>% 
  print(target = "/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/PAPER/02_review/table/Table 3.docx")

#



# numbers abstract --------------------------------------------------------


dat_seasonal_gls[variable == "meanHS_DJF" & term == "year0"] %>% summary

dat_seasonal_gls[term == "year0",
                 .(decadal_trend = mean(estimate*10)),
                 .(variable)] 




# numbers conclusion ------------------------------------------------------

dat_month_gls[term == "year0", 
              table(trend_pos = estimate > 0)] %>% 
  prop.table()

dat_month_gls[term == "year0", 
              table(trend_pos = estimate > 0, p_sig = p.value < 0.05)] %>% 
  prop.table()

dat_month_gls[term == "year0", 
              table(trend_pos = estimate > 0, p_sig = p.value < 0.05)] %>% 
  prop.table(1)
