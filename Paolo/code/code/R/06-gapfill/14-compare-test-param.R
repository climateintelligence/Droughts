# visualize gapfill cross validation results
# - 



library(data.table)
library(magrittr)
library(lubridate)
library(ggplot2)
library(patchwork)
library(scico)
library(forcats)
library(fs)
library(officer)
library(flextable)
library(foreach)



# read data ---------------------------------------------------------------


# meta
dat_meta <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/03_QC1/rds/1961-2020/meta_long_HN_HS.rds")

# param test
all_files <- dir_ls("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/04_GAPFILL/rds/cv-test-param/")

dat_cv <- foreach(
  i_fn = all_files,
  .final = rbindlist
) %do% {
  
  dat <- readRDS(i_fn)
  dat[, value_diff := value_fill - value_true]
  
  i_fn %>% 
    path_file %>% 
    path_ext_remove -> fn_id

  dat[, ff := fn_id]
  dat  
}


# add cv5day orig
dat_5day <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/04_GAPFILL/rds/cv-02-5day.rds")
dat_5day[, value_diff := value_fill - value_true]
dat_5day[, ff := "00_original"]

dat_cv2 <- rbindlist(list(dat_cv, dat_5day), use.names = T)

# prep data ---------------------------------------------------------------

dat_cv_summ <- dat_cv2[!is.na(i_rep) & !is.na(value_fill),
                      .(mu_diff = mean(value_diff),
                        mu_true = mean(value_true),
                        mu_true_nozero = mean(value_true[value_true != 0]),
                        sd = sd(value_diff),
                        mae = mean(abs(value_diff)),
                        mae_nozero = mean(abs(value_diff[value_true != 0])),
                        rmse = sqrt(mean(value_diff*value_diff)),
                        n_rep = .N),
                      .(ff, Name, month)]

dat_cv_summ %>% 
  merge(dat_meta, by = "Name") -> dat_cv_summ
dat_cv_summ[, country := substr(Provider, 1, 2)]
dat_cv_summ[, month_f := factor(month.abb[month], levels = month.abb[c(10:12, 1:9)])]
dat_cv_summ[, elev_f := cut(Elevation, 
                            breaks = c(0, 500, 1000, 1500, 2000, 3000),
                            dig.lab = 4,
                            include.lowest = F)]
dat_cv_summ %>% summary
with(dat_cv_summ, table(month_f, country))


dat_cv_summ[, elev_f2 := cut(Elevation, 
                         breaks = c(0, 1000, 2000, 3000),
                         dig.lab = 4,
                         include.lowest = F)]



# plot --------------------------------------------------------------------

dat_plot <- dat_cv_summ[n_rep > 50]

# manual boxplot.stats (to remove outliers)

dat_plot[ff == "00_original", 
         .(Name, month,
           orig_mae = mae, orig_mae_nozero = mae_nozero)] %>% 
  merge(dat_plot[ff != "00_original"],
        by = c("Name", "month")) -> dat_plot2

dat_plot2[, diff_mae := mae - orig_mae]

dat_plot_manual <- dat_plot2[, 
                            .(stats = boxplot.stats(diff_mae)$stats, 
                              xx = c("ymin", "lower", "middle", "upper", "ymax")),
                            .(ff, elev_f2, month_f)]
# boxplot.stats(dat_plot$mae) %>% str

dat_plot_manual %>% 
  dcast(... ~ xx, value.var = "stats") %>% 
  ggplot(aes(ff, ymin = ymin, lower = lower, middle = middle, upper = upper, ymax = ymax))+
  geom_boxplot(stat = "identity")+
  geom_hline(yintercept = 0, colour = "grey")+
  facet_grid(elev_f2 ~ month_f, scales = "free_y")+
  theme_bw()+
  theme(axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5))+
  xlab(NULL)+
  ylab("MAE [cm]")


# -> not so much difference. method works rather well, our choices were sensible
# small benefits expected for increased corr and weighting by corr...
# but only for some cases ... and probably not worth it...
