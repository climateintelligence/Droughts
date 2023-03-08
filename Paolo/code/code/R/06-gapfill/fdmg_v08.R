# changes made
# --------------------------- #



# ------- v01 --------------- #  
# - renamed input parameters
# - removed plot
# ------- v02 --------------- #  
# - rewrote distance calculation using geosphere package 
#    -> slightly different weights because radius more digits
#    -> minimal different results
# ------- v03 --------------- #  
# - rewrote stns_to_fill parameter (which stations)
# - added separated dates vectors
# - added verbose parameter (0: nothing, 1:progress, 2:much detail)
# - improved finding the window data (~1/4 time saved)
# ------- v04b -------------- #  
# - added pre-subset stations within horiz and vertical limits
# - added new param max_dist_vert_m
# - rewrote parameter calculation (using matrix)
# ------- v05 --------------- #  
# - added save_ref_parameters option
# - splitted input into data (time series) and dates, with two options to supply dates
# - remove parameter "variable" and instead put "ratio_var" and "wetdays"
# ------- v06 --------------- #  
# - return list(series, fillcodes) instead of global assignment / attribute
# - fixed feb 29 issue, so also years without feb 29 are considered for window
# - added n_ref_min parameter (n_ref changed to n_ref_max)
# - added some input checks
# - added sort_by and weight_by parameters (and weight_by_extra storing tau halving distances)
# ------- v07 --------------- #  
# - added roxygen documentation
# ------- v08 --------------- #  
# - added rows_to_fill parameter


# requirements
# - df_meta[, c("name", "long", "lat", "elev")]
# - sort_by = c("corr", "dist_v", "dist_h")

# packages needed
library(lubridate)
library(data.table)
library(geosphere)




# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #

#' Gapfilling of daily meteorological time series based on spatial neighbours
#'
#' @param df_meta data.frame containing meta information on stations. 
#'                Should have columns named c("name", "long", "lat", "elev")
#' @param mat_series matrix of meteorological series. 
#'                   Each column is a station, ncol(mat_series) should be equal nrow(df_meta).
#'                   Order of stations in the columns of mat_series should be identical 
#'                   to the order rows in df_meta,
#' @param mat_ymd matrix of dimension nrow(mat_series) by 3. 
#'                Column should contain the year, month, and day (in that order) of the series.
#'                Instead of this, vec_dates could alternatively be supplied.
#' @param vec_dates vector of class Date, associated to the series. 
#'                  Length should be equal to nrow(mat_series).
#'                  Instead of this, mat_ymd could alternatively be supplied.
#' @param stns_to_fill numeric, stations which should be filled, if NULL, all stations are selected
#' @param rows_to_fill numeric, rows which should be filled, if NULL, all missing values are filled
#' @param min_corr numeric, threshold of minimum correlation for stations to be considered
#' @param elev_threshold numeric or NULL, elevation threshold, if not NULL, then if the gap series
#'                       has elevation below, then only neighbouring stations also below are considered.
#' @param max_dist_horiz_km numeric, maximum horizontal distance in km for stations to be considered.
#' @param max_dist_vert_m NULL or numeric, if not NULL, maximum vertical distance in m for stations to be considered. 
#' @param frac_ref_window numeric between 0 and 1, fraction of data which gap and neighbouring series 
#'                        should have in common
#' @param n_ref_max numeric, maximum number of reference stations
#' @param n_ref_min numeric, minimum number of reference stations
#' @param digits_round numeric, number of digits to round the gapfilled value
#' @param ratio_var logical, if TRUE scaling factors are multiplicative, if FALSE additive
#' @param sort_by character, one of c("dist_h", "dist_v, "corr"), sorting criteria for reference stations
#' @param weight_by character, one of c("dist_h", "dist_v, "corr"), weighting criteria for reference stations
#' @param weight_by_extra list of length 3, containing the halving distances for the weighting,
#'                        list elements should have names tau_h, tau_v, tau_corr.
#'                        e.g. list(tau_h = 50, tau_v = 300, tau_corr = 0.3)
#' @param min_days_around_gap minimum number of valid data required in gap series around gap
#' @param window_hw_years half-width of year window, e.g. if 10, 10 years before and 10 years after the
#'                        gap are tested for finding suitable data
#' @param window_hw_days_min half-width of day window, e.g. if 15, a 31 day-window around the gap is searched
#'                           for available data. The window is increased by 5 until min_days_around_gap
#'                           is fullfilled. The window is only increased of to a maximum of window_hw_days_max.
#' @param window_hw_days_max see window_hw_days_min.
#' @param verbose numeric, if 0 no information is printed, if 1, some progress info is printed.
#' @param save_ref_parameter NULL or character, if not NULL, should contain a valid folder path,
#'                           to which information of the calculated reference station values 
#'                           (such as correlation, distances, etc.) per gap are stored.
#' @param wetdays logical of numeric, if not FALSE, then the number of wetdays above the threshold 
#'                (1 for TRUE, otherwise the numeric value of wetdays) are calculated.
#'
#' @return a list with two elements named (mat_series_filled, mat_fillcodes)
#'         containing the filled mat_series and a matrix of the same dimensions holding 
#'         the fill code (1 for original values, 222 for reconstructed gaps). 
#'         If stns_to_fill is not NULL, then only these stations are returned.
#' @export
#'
#' @examples
fill_daily_meteo_gaps <- function(df_meta, 
                                  mat_series, 
                                  mat_ymd, 
                                  vec_dates, 
                                  stns_to_fill = NULL, 
                                  rows_to_fill = NULL,
                                  min_corr = 0.7, 
                                  elev_threshold = NULL, 
                                  max_dist_horiz_km = 200,
                                  max_dist_vert_m = 500,
                                  frac_ref_window = 0.8, 
                                  n_ref_max = 5, 
                                  n_ref_min = 1,
                                  digits_round = 0,
                                  ratio_var = T,
                                  sort_by = "corr", 
                                  weight_by = "dist_v",
                                  weight_by_extra = list(tau_h = 50, tau_v = 250, tau_corr = 0.3),
                                  min_days_around_gap = 150, 
                                  window_hw_years = 10,
                                  window_hw_days_min = 15, 
                                  window_hw_days_max = 46,
                                  verbose = 1, # other
                                  save_ref_parameter = NULL,
                                  wetdays = F){
  

# setup -------------------------------------------------------------------

  
  n_stn <- nrow(df_meta)
  
  # check input
  if(n_stn != ncol(mat_series)) stop("Number of stations in df_meta (rows) and mat_series (cols) not equal")
  if(!missing(mat_ymd) && nrow(mat_ymd) != nrow(mat_series)) stop("Date mismatch: Number of rows in mat_series and mat_ymd unequal")
  if(!missing(vec_dates) && length(vec_dates) != nrow(mat_series)) stop("Date mismatch: Length of vec_dates not equal to number of rows in mat_series")
  
  # distance matrix
  mat_dist_km <- geosphere::distm(df_meta[, c("long", "lat")], fun = geosphere::distCosine) / 1000
  # matrix of station distance weights
  mat_dist_weight <- exp( -(mat_dist_km^2) / (weight_by_extra$tau_h^2/log(2)) ) 
  
  
  # separate date and series
  if(missing(mat_ymd) & !missing(vec_dates)){
    years <- year(vec_dates)
    months <- month(vec_dates)
    days <- day(vec_dates)
    dates <- vec_dates
  } else if(!missing(mat_ymd)){
    years <- mat_ymd[, 1]
    months <- mat_ymd[, 2]
    days <- mat_ymd[, 3]
    dates <- make_date(years, months, days)
  } else stop("One of mat_ymd or vec_dates has to be supplied.")
  
  years_min <- min(years)
  years_max <- max(years)
  
  # creat out matrices
  mat_series_filled <- mat_series #matrix where filled series are stored
  mat_fillcodes <- mat_series #matrix of codes = 1 for original values, 222 for reconstructed gaps
  mat_fillcodes[!is.na(mat_fillcodes)] <- 1

  # which stations
  if(is.null(stns_to_fill)) stns_to_fill <- 1:n_stn
  

# start loop of stations --------------------------------------------------

  for(i_stn in stns_to_fill){
    
    if(verbose > 0) cat("Started station", i_stn, "of", n_stn, ":", colnames(mat_series)[i_stn], "\n")
    
    # init possibly save_ref_parameter
    if(!is.null(save_ref_parameter)){
      outfile <- file.path(save_ref_parameter, paste0(df_meta$name[i_stn], ".csv"))
      l_out_save <- list()
    }
    

# start loop rows (missing dates) -----------------------------------------

    if(is.null(rows_to_fill)) rows_to_fill <- which(is.na(mat_series[, i_stn])) 
    n_filled <- 0
    for(i_row in rows_to_fill){
      
      n_filled <- n_filled + 1
      if(verbose > 0 & n_filled %% 100 == 0) cat("  filled", n_filled, "of", length(rows_to_fill), "\n")
      
      i_date <- dates[i_row]
      

# get gap series window data ----------------------------------------------

      
      
      # get available years in window around i_date
      current_window_years <- seq(years[i_row] - window_hw_years, years[i_row] + window_hw_years)
      current_window_years <- current_window_years[current_window_years >= years_min & current_window_years <= years_max]
      
      # get all dates/indices in window (days * years)
      for(current_window_hw_days in seq(window_hw_days_min, window_hw_days_max, by = 5)){
        
        l_window <- lapply(current_window_years, function(i_year){
          
          if(month(i_date) == 2 & day(i_date) == 29){
            i_date_seq <- make_date(i_year, month(i_date), 28) 
          } else {
            i_date_seq <- make_date(i_year, month(i_date), day(i_date))
          }
          seq(i_date_seq - current_window_hw_days, 
              i_date_seq + current_window_hw_days, 
              by = "day")
          
        })
        
        current_window_dates <- do.call("c", l_window)
        
        current_window_ind <- match(current_window_dates, dates)
        current_window_ind <- current_window_ind[!is.na(current_window_ind)]
        values_gap_series <- mat_series[current_window_ind, i_stn]
        
        # break out of loop, if enough values
        if(sum(!is.na(values_gap_series)) > min_days_around_gap) break
        
      }
      
      # skip if not enogh values in gap series
      if(sum(!is.na(values_gap_series)) < min_days_around_gap) next
      

# identify candidates for reference series --------------------------------


            
      # subset to nearby stations within horizontal and vertical limits
      # preselect candidate reference series
      
      # horizontal limits
      ref_stns_possible <- which(mat_dist_km[i_stn, ] < max_dist_horiz_km)
      # not gap series
      ref_stns_possible <- ref_stns_possible[ref_stns_possible != i_stn]
      
      # elev_threshold (if gap series below, only stns below considered)
      if(!is.null(elev_threshold)){
        if(df_meta$elev[i_stn] < elev_threshold) {
          ref_stns_possible <- intersect(ref_stns_possible, 
                                         which(df_meta$elev < elev_threshold))
        }
      }
      
      # vertical limits
      if(!is.null(max_dist_vert_m)){
        ref_stns_possible <- intersect(ref_stns_possible, 
                                       which(abs(df_meta$elev - df_meta$elev[i_stn]) < max_dist_vert_m))
      }
      
      # not NA at gap date
      ref_stns_possible <- intersect(ref_stns_possible,
                                     which(!is.na(mat_series[i_row,])))
      

# get fill parameters for reference series --------------------------------

      
      
      # get parameters of possible series
      mat_fill_param <- matrix(nrow = length(ref_stns_possible), ncol = 11)
      
      for(i_fill in seq_along(ref_stns_possible)){
        
        i_ref <- ref_stns_possible[i_fill]
        values_ref <- mat_series[current_window_ind, i_ref]
        
        # lgl common data
        lgl_common <- !is.na(values_gap_series) & !is.na(values_ref)
        
        # do nothing if not enough data in common
        if(sum(lgl_common) < min_days_around_gap*frac_ref_window) next
        
        # get common data
        values_gap_series_common <- values_gap_series[lgl_common]
        values_ref_common <- values_ref[lgl_common]
        
        # correlations
        # make correlation just above threshold, if both series completey 0 -> so 0 periods get filled too (e.g. summer snow)
        if(all(values_gap_series_common == 0) & all(values_ref_common == 0)){
          correlation <- min_corr + 0.001 
          # make 0 cor if either stn or ref is all 0 (both is checked before) -> sd=0 in cor, so not working
        } else if(all(values_gap_series_common == 0) | all(values_ref_common == 0)){
          correlation <- 0 
          # otherwise normal cor
        } else {
          correlation <- cor(values_gap_series_common, values_ref_common, method="pearson")
        }
        
        # number of wet days
        if(wetdays){
          n_wetdays <- sum(values_gap_series_common > wetdays & values_ref_common > wetdays)
        } else {
          n_wetdays <- NA
        }
        
        
        mean_gap <- mean(values_gap_series_common)
        mean_ref <- mean(values_ref_common)
        
        
        if(mean_gap == 0 & mean_ref == 0){
          # scaling factor 0 if both series completely 0
          conv_fact <- 0 
        } else if(ratio_var){
          conv_fact <- mean_gap / mean_ref
        } else {
          conv_fact <- mean_gap - mean_ref
        }
        
        
        mat_fill_param[i_fill, ] <- c(i_stn,
                                      df_meta$elev[i_stn],
                                      i_ref,
                                      df_meta$elev[i_ref],
                                      mat_dist_km[i_stn, i_ref],
                                      mat_dist_weight[i_stn, i_ref],
                                      df_meta$elev[i_ref] - df_meta$elev[i_stn],
                                      correlation,
                                      sum(lgl_common),
                                      conv_fact,
                                      n_wetdays)       
        
      }
      
      colnames(mat_fill_param) <- c("ind_gap", "elev_gap", "ind_ref", "elev_ref",
                                    "dist_h_km", "dist_h_weight", "dist_v_m", "corr", 
                                    "n_common", "conv_fact", "n_wetdays")
      
      rownames(mat_fill_param) <- df_meta$name[ref_stns_possible]


# select the final reference series ---------------------------------------

      # remove NA (not reached the threshold for common data) and skip if empty
      mat_fill_param <- mat_fill_param[!is.na(mat_fill_param[, "ind_ref"]), , drop = F]
      if(nrow(mat_fill_param) == 0) next
      
      # selection of the n_ref stations that will be used 
      mat_selected_ref <- mat_fill_param[mat_fill_param[, "corr"] >= min_corr, , drop = F]
      
      # skip if none
      if(nrow(mat_selected_ref) == 0) next
      
      # sort by
      if(sort_by == "corr"){
        ord <- order(mat_selected_ref[, "corr"], decreasing = T)
      } else if(sort_by == "dist_h"){
        ord <- order(mat_selected_ref[, "dist_h_km"], decreasing = F)
      } else if(sort_by == "dist_v") {
        ord <- order(abs(mat_selected_ref[, "dist_v_m"]), decreasing = F)
      } else stop('sort_by must be in c("corr", "dist_h", "dist_v")')
      mat_selected_ref_ordered <- mat_selected_ref[ord, , drop = F]
    
      
      # select stations
      n_ref_actual <- min(n_ref_max, nrow(mat_selected_ref_ordered))
      # skip if too few
      if(n_ref_actual < n_ref_min) next
      # else use the available
      i_selected_ref <- mat_selected_ref_ordered[1:n_ref_actual, "ind_ref"]
      
      # save ref parameters
      if(!is.null(save_ref_parameter)){
        l_out_save[[as.character(i_date)]] <- data.table::as.data.table(mat_selected_ref_ordered,
                                                                        keep.rownames = "name_ref")
      }
      
      # weigthed means
      wm_values <- mat_series[i_row, i_selected_ref]
      wm_conv_fact <- mat_selected_ref_ordered[1:n_ref_actual, "conv_fact"]
      
      # weight by
      if(weight_by == "corr"){
        xx <- mat_selected_ref_ordered[1:n_ref_actual, "corr"]
        wm_weights <- exp( -(1 - xx^2) / (weight_by_extra$tau_corr^2/log(2)) ) 
      } else if(weight_by == "dist_h"){
        wm_weights <- mat_selected_ref_ordered[1:n_ref_actual, "dist_h_weight"]
      } else if(weight_by == "dist_v") {
        xx <- mat_selected_ref_ordered[1:n_ref_actual, "dist_v_m"]
        wm_weights <- exp( -(xx^2) / (weight_by_extra$tau_v^2/log(2)) ) 
      } else stop('weight_by must be in c("corr", "dist_h", "dist_v")')
      

      
      # fill value
      if(ratio_var){
        fill_value <- weighted.mean(wm_values * wm_conv_fact, wm_weights)
      } else { 
        fill_value <- weighted.mean(wm_values + wm_conv_fact, wm_weights)
      }
      
      mat_series_filled[i_row, i_stn] <- round(fill_value, digits = digits_round)
      mat_fillcodes[i_row, i_stn] <- 222
      
    }#closing iter on gaps
    
    # print parameter info
    if(!is.null(save_ref_parameter)){
      dt_out_save <- data.table::rbindlist(l_out_save, idcol = "date")
      if(nrow(dt_out_save) == 0) dt_out_save <- data.table::data.table(no_ref_station_found = 0)
      data.table::fwrite(dt_out_save, outfile)
    }
    
  }#closing iter on stations
  
  #only columns corresponding to tested series are returned 
  list(
    mat_series_filled = mat_series_filled[, c(stns_to_fill)],
    mat_fillcodes = mat_fillcodes[, c(stns_to_fill)]
  )
}
