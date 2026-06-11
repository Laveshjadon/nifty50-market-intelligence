-- 1. Remove Non-Market Hours (Before 09:15 and After 15:30)
DELETE FROM min_5 
WHERE bucket_time::time < '09:15:00' 
   OR bucket_time::time > '15:30:00';


DELETE FROM min_5 
WHERE EXTRACT(DOW FROM bucket_time) IN (0, 6);

-- Apply the same cleaning logic to the index table
DELETE FROM nifty50_index_5m 
WHERE bucket_time::time < '09:15:00' 
   OR bucket_time::time > '15:30:00';

DELETE FROM nifty50_index_5m 
WHERE EXTRACT(DOW FROM bucket_time) IN (0, 6);


-- 2. Align all Nifty 50 stocks to the same common start date.
-- LTIM is the latest starter in the current universe, so trimming older rows
DELETE FROM min_5
WHERE bucket_time < '2016-07-21 09:40:00';

DELETE FROM nifty50_index_5m
WHERE bucket_time < '2016-07-21 09:40:00';


CREATE INDEX IF NOT EXISTS idx_min5_composite ON min_5 (ticker, bucket_time);
CREATE INDEX IF NOT EXISTS idx_index_composite ON nifty50_index_5m (index_name, bucket_time);


VACUUM ANALYZE min_5;
VACUUM ANALYZE nifty50_index_5m;
