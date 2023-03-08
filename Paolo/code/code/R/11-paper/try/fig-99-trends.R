# plot trends


library(data.table)
library(magrittr)
library(lubridate)
library(ggplot2)
library(scico)

library(forcats)
library(foreach)
library(flextable)
library(officer)

dat_meta <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/05_MONTHLY/rds/1961-2020_gapfilled_HS_sub/meta_long_HS.rds")
dat_meta_clust <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/PAPER/rds/meta-with-cluster-01.rds")

# choose hydro_year or calendar year
dat_lm <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/PAPER/rds/trends-01-mw-hydro-year.rds")
dat_lm  <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/PAPER/rds/trends-02-mw-calendar-year.rds")

# dat_lm_sub <- dat_lm[term == "year0" & mw_year_start %in% c(1961, 1976, 1990)]
# dat_lm_sub <- dat_lm[term == "year0" & mw_year_start %in% c(1961, 1971, 1981, 1990)]

dat_lm_full  <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/PAPER/rds/trends-03-full_1971-2019-calyear.rds")

# try scatter plots ---------------------------------------------------------------

dat_lm_sub <- dat_lm[!is.na(statistic) & 
                       term == "year0" & 
                       mw_year_start %in% c(1961, 1966, 1971, 1976, 1981, 1986)]

dat_plot <- merge(dat_lm_sub, dat_meta_clust, by = "Name")
mitmatmisc::add_month_fct(dat_plot, 10)
dat_plot[, mw_year_fct := paste0(mw_year_start,"-",mw_year_end)]

dat_plot %>% 
  ggplot(aes(Elevation, estimate, colour = cluster_fct))+
  geom_point(size = 0.4)+
  facet_grid(month_fct ~ mw_year_fct)+
  theme_bw()+
  scale_color_brewer(palette = "Set1")

# supplement: table with constant station set? needed?
# not really, there is afterwards the complete period stuff!
dat_lm[term == "year0", .N, .(month, mw_year_start)] %>% 
  ggplot(aes(mw_year_start, N))+
  geom_point()+
  facet_wrap(~month)


# average trend? ----------------------------------------------------------

dat_lm[!is.na(statistic) & term == "year0" & mw_year_start < 1990] %>% 
  merge(dat_meta, by = "Name") -> dat_avg_trend

# subset to common stations to all period
dat_avg_trend[, .N, .(month, Name)] %>% summary
dat_avg_trend[, .N, .(month, Name)] %>% 
  .[N == 29] %>% 
  merge(dat_avg_trend, by = c("month", "Name")) -> dat_avg_trend

mitmatmisc::add_month_fct(dat_avg_trend, 10)
dat_avg_trend[, mw_year_fct := paste0(mw_year_start,"-",mw_year_end)]

dat_avg_trend$Elevation %>% qplot
dat_avg_trend$Elevation %>% summary

# dat_avg_trend[, elev_fct := cut(Elevation, breaks = seq(0, 3000, by = 100), dig.lab = 5)]
dat_avg_trend[, elev_fct := cut(Elevation, breaks = seq(0, 3000, by = 250), dig.lab = 5)]
# dat_avg_trend[, elev_fct := cut_number(Elevation, 30, dig.lab = 5)]

# dat_avg_trend[, .N, .(month, elev_fct, mw_year_start)] %>% 
#   ggplot(aes(mw_year_start, N, colour = as.numeric(elev_fct)))+
#   geom_line(aes(group = elev_fct))+
#   facet_wrap(~ month)


dat_avg_trend %>% 
  merge(dat_lm[term == "(Intercept)", .(Name, month, y0 = estimate, mw_year_start)],
        by = c("Name", "month", "mw_year_start")) -> dat_avg_trend2

dat_avg_trend2[, estimate_relative := estimate / y0]
dat_avg_trend2 %>% summary
dat_avg_trend2[is.infinite(estimate_relative)]
dat_avg_trend2[y0 != 0] %>% summary
dat_avg_trend2[abs(estimate_relative) > 10]
dat_avg_trend2[abs(y0) > 0.1] %>% summary

dat_plot_avg <- dat_avg_trend2[abs(y0) > 0.1, 
                               .(nn = .N,
                                 mean_trend = mean(estimate),
                                 mean_trend_relative = mean(estimate/y0)),
                               .(mw_year_fct, month_fct, elev_fct)]
summary(dat_plot_avg)
dat_plot_avg

dat_plot_avg %>% 
  ggplot(aes(mw_year_fct, elev_fct, fill = mean_trend))+
  geom_tile()+
  facet_wrap(~month_fct, nrow = 2)+
  scale_fill_scico(palette = "vik", 
                   limits = c(-1, 1) * max(abs(range(dat_plot_avg$mean_trend))), 
                   direction = -1)+
  theme_bw()


dat_plot_avg %>% 
  ggplot(aes(mw_year_fct, elev_fct, fill = nn))+
  geom_tile()+
  facet_wrap(~month_fct, nrow = 2)+
  scale_fill_binned(type = "viridis", n.breaks = 10)+
  theme_bw()

dat_plot_avg[month_fct == "Dec"] %>% with(table(elev_fct, nn))
# -> not many stations at high elev !? < 10

# sub for only same stations


# average over period
dat_plot_avg_full <- dat_avg_trend2[abs(y0) > 0.1, 
                                    .(nn = .N,
                                      mean_trend = mean(estimate),
                                      mean_trend_relative = mean(estimate/y0)),
                                    .(month_fct, elev_fct)]
summary(dat_plot_avg_full)

dat_plot_avg_full %>% 
  ggplot(aes(mean_trend, elev_fct))+
  geom_vline(xintercept = 0)+
  geom_point()+
  facet_wrap(~month_fct, nrow = 2)+
  theme_bw()
  

# average over period, but keep stns
dat_plot_avg_full2 <- dat_avg_trend2[abs(y0) > 0.1, 
                                    .(nn = .N,
                                      mean_trend = mean(estimate),
                                      mean_trend_relative = mean(estimate/y0)),
                                    .(month_fct, elev_fct, Name)]
summary(dat_plot_avg_full2)

dat_plot_avg_full2[nn == 29] %>% 
  ggplot(aes(mean_trend, elev_fct))+
  geom_vline(xintercept = 0)+
  geom_jitter(width = 0, height = 0.3)+
  facet_wrap(~month_fct, nrow = 2)+
  theme_bw()




# relative?
dat_plot_avg %>% 
  ggplot(aes(mw_year_fct, elev_fct, fill = mean_trend_relative))+
  geom_tile()+
  facet_wrap(~month_fct, nrow = 2)+
  scale_fill_scico(palette = "vik",
                   limits = c(-1, 1) * max(abs(range(dat_plot_avg$mean_trend_relative))), 
                   direction = -1)+
  theme_bw()



# -> need some summary information! 
# maybe both, one over space (but with mw_year), then over all period
# for supplement: do analysis with reduced data (only common stations)
# maybe do some clustering of regions based on PCA?



# full period trends ------------------------------------------------------


dat_plot_full <- dat_lm_full[!is.na(statistic) & term == "year0"] %>% merge(dat_meta_clust, by = "Name")
dat_plot_full[, est_low := estimate - 1.96 * std.error]
dat_plot_full[, est_high := estimate + 1.96 * std.error]
mitmatmisc::add_month_fct(dat_plot_full, 10)

dat_plot_full


dat_plot_full %>% 
  ggplot(aes(estimate, Elevation, xmin = est_low, xmax = est_high, colour = cluster_fct))+
  geom_vline(xintercept = 0)+
  geom_pointrange(size = 0.2, fatten = 1)+
  # geom_jitter(width = 0, height = 0.3)+
  # facet_wrap(~month_fct, nrow = 2)+
  facet_grid(cluster_fct ~ month_fct, scales = "free_x", space = "free")+ # free_x or free
  theme_bw()+
  scale_color_brewer(palette = "Set1")

dat_plot_full[, elev_fct := cut(Elevation, breaks = seq(0, 3000, by = 250), dig.lab = 5)]
dat_plot_full[, trend_sign := ifelse(estimate <= 0, "-", "+")]


# -> with regions: need a table for average trend and % sig (or second plot)



dat_plot_full_sig <- dat_plot_full[, 
                                   .N, 
                                   .(month_fct, elev_fct, p_sig = p.value < 0.05, trend_sign)]
dat_plot_full_sig[, perc := N / sum(N), .(month_fct, elev_fct)]
setorder(dat_plot_full_sig, month_fct, elev_fct, p_sig)
dat_plot_full_sig





dat_add_sig <- dat_plot_full_sig[p_sig == "TRUE"]
dat_add_sig[, perc_formatted := scales::percent(perc, .1)]
dat_add_sig[, lbl := paste0("N=", N, " / ", perc_formatted)]
dat_add_sig[, lbl := paste0("N=", N, " (", perc_formatted, ")")]
dat_add_sig[, lbl_ymin := as.numeric(elev_fct)*250 - 250]
dat_add_sig[, lbl_ymax := as.numeric(elev_fct)*250]
dat_add_sig[, lbl_x := ifelse(trend_sign == "-", 
                              min(dat_plot_full$est_low), 
                              max(dat_plot_full$est_high))]
dat_add_sig[, lbl_hjust := ifelse(trend_sign == "-", 0, 1)]
dat_add_sig[, seg_xend_sig := ifelse(trend_sign == "-", 1, -1)]


dat_plot_full_avg <- dat_plot_full[, 
                                   .(mean_trend = mean(estimate)),
                                   .(month_fct, elev_fct)]
dat_plot_full_avg[, ymin := as.numeric(elev_fct)*250 - 250]
dat_plot_full_avg[, ymax := as.numeric(elev_fct)*250]


dat_plot_full %>% 
  ggplot(aes(estimate, Elevation, xmin = est_low, xmax = est_high))+
  geom_vline(xintercept = 0)+
  geom_pointrange(size = 0.2, fatten = 1, colour = "grey20")+
  
  geom_point(inherit.aes = F, data = dat_plot_full_avg,
             aes(x = mean_trend, y = (ymin+ymax)/2), colour = "red")+
  
  geom_segment(inherit.aes = F, data = dat_add_sig,
               aes(x = lbl_x, xend = lbl_x + 2 * seg_xend_sig, y = lbl_ymin, yend = lbl_ymin),
               linetype = "dashed", alpha = 0.5)+
  geom_segment(inherit.aes = F, data = dat_add_sig,
               aes(x = lbl_x, xend = lbl_x + 2 * seg_xend_sig, y = lbl_ymax, yend = lbl_ymax),
               linetype = "dashed", alpha = 0.5)+
  geom_text(inherit.aes = F, data = dat_add_sig,
             aes(x = lbl_x, y = (lbl_ymin + lbl_ymax)/2, hjust = lbl_hjust, label = lbl),
             vjust = 0.5, size = 3, alpha = 0.5)+
  
  facet_wrap(~month_fct, nrow = 2)+
  theme_bw()+
  scale_y_continuous(minor_breaks = seq(0, 3000, by = 250))

# nice! maybe add a small map of the station locations

# table of histogram ------------------------------------------------------

dat_table <- copy(dat_plot)

dat_table$estimate %>% qplot
dat_table$estimate %>% summary
dat_table[, trend_bins := cut(estimate, breaks = -10 : 6)]
dat_table[, .N, keyby = trend_bins]

dat_table[, trend_bins := cut(estimate, breaks = c(-10, -4:4, 6), right = F)]

dat_table2 <- dat_table[, .(nn = .N), .(trend_bins, month_fct, mw_year_fct)]
dat_table2[, perc := nn/sum(nn), .(month_fct, mw_year_fct)]

dat_table3 <- dat_table2 %>% 
  dcast(month_fct + trend_bins ~ mw_year_fct, value.var = list("nn", "perc"))

# dat_table3[, trend_bins := as.character(trend_bins)]
setorder(dat_table3, month_fct, -trend_bins)

dat_header <- data.table(col_keys = colnames(dat_table3))
dat_header[, header2 := tstrsplit(col_keys, "_", keep = 1)]
dat_header[header2 == "nn", header2 := "# count"]
dat_header[header2 == "perc", header2 := "Percent"]
dat_header[c(1,2) , header2 := ""]
dat_header[, header := tstrsplit(col_keys, "_", keep = 2)]
dat_header[c(1,2), header := c("Month", "Trend [cm/yr]")]


dat_table3 %>% 
  flextable() %>% 
  set_header_df(dat_header) %>% 
  merge_h(part = "header") %>% 
  theme_booktabs() %>% 
  colformat_int(na_str = "",
                j = c("nn_1961-1990", "nn_1971-2000", "nn_1981-2010", "nn_1990-2019")) %>% 
  set_formatter(`perc_1961-1990` = scales::percent_format(1),
                `perc_1971-2000` = scales::percent_format(1),
                `perc_1981-2010` = scales::percent_format(1),
                `perc_1990-2019` = scales::percent_format(1)) %>%
  merge_v() %>% 
  valign(valign = "top") %>% 
  fix_border_issues() %>% 
  autofit() 
