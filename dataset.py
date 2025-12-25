import json
import math
from pathlib import Path
import numpy as np
from tqdm.notebook import tqdm

from config import *
from utils import *


# =========================
# STREAMS
# =========================
class TradeStream:
    def __init__(self, files, t_start_ms: int, t_end_ms: int):
        self.files = files
        self.t_start_ms = int(t_start_ms)
        self.t_end_ms = int(t_end_ms)
        self.fi = -1
        self.fh = None
        self.reader = None
        self.buf = None
        self._open_next_file()

    def _close(self):
        if self.fh is not None:
            self.fh.close()
        self.fh = None
        self.reader = None

    def _open_next_file(self):
        self._close()
        self.fi += 1
        if self.fi >= len(self.files):
            return False
        self.fh = self.files[self.fi].open("r", newline="")
        self.reader = csv.reader(self.fh)
        return True

    def _read_next_row(self):
        while True:
            if self.reader is None:
                return None
            try:
                row = next(self.reader)
            except StopIteration:
                ok = self._open_next_file()
                if not ok:
                    return None
                continue
            if not row:
                continue
            if row[0].strip().lower() == "id":
                continue
            return row

    def peek(self):
        while self.buf is None:
            row = self._read_next_row()
            if row is None:
                return None
            try:
                ts = int(row[4])
            except Exception:
                continue
            if ts < self.t_start_ms:
                continue
            if ts > self.t_end_ms:
                return None
            self.buf = row
        return self.buf

    def pop(self):
        row = self.peek()
        self.buf = None
        if row is None:
            return None
        ts = int(row[4])
        price = float(row[1])
        qty = float(row[2])
        quote_qty = float(row[3])
        is_buyer_maker = row[5].strip().lower() == "true"
        return ts, price, qty, quote_qty, is_buyer_maker


class BookStream:
    def __init__(self, files, t_start_ms: int, t_end_ms: int):
        self.files = files
        self.t_start_ms = int(t_start_ms)
        self.t_end_ms = int(t_end_ms)
        self.fi = -1
        self.fh = None
        self.reader = None
        self.buf_row = None
        self._open_next_file()

    def _close(self):
        if self.fh is not None:
            self.fh.close()
        self.fh = None
        self.reader = None

    def _open_next_file(self):
        self._close()
        self.fi += 1
        if self.fi >= len(self.files):
            return False
        self.fh = self.files[self.fi].open("r", newline="")
        self.reader = csv.reader(self.fh)
        return True

    def _read_next_row(self):
        while True:
            if self.reader is None:
                return None
            try:
                row = next(self.reader)
            except StopIteration:
                ok = self._open_next_file()
                if not ok:
                    return None
                continue
            if not row:
                continue
            if row[0].strip().lower() == "timestamp":
                continue
            return row

    def _get_row(self):
        if self.buf_row is not None:
            row = self.buf_row
            self.buf_row = None
            return row
        return self._read_next_row()

    def next_snapshot(self):
        while True:
            row = self._get_row()
            if row is None:
                return None
            ts = book_ts_to_ms(row[0].strip())
            if ts < self.t_start_ms:
                continue
            if ts > self.t_end_ms:
                return None
            break

        bid_depth = np.zeros(5, dtype=np.float32)
        ask_depth = np.zeros(5, dtype=np.float32)
        bid_notional = np.zeros(5, dtype=np.float32)
        ask_notional = np.zeros(5, dtype=np.float32)

        while True:
            p = int(float(row[1]))
            depth = float(row[2])
            notional = float(row[3])

            if p < 0:
                idx = abs(p) - 1
                if 0 <= idx < 5:
                    bid_depth[idx] = depth
                    bid_notional[idx] = notional
            else:
                idx = p - 1
                if 0 <= idx < 5:
                    ask_depth[idx] = depth
                    ask_notional[idx] = notional

            row2 = self._read_next_row()
            if row2 is None:
                break
            ts2 = book_ts_to_ms(row2[0].strip())
            if ts2 != ts:
                self.buf_row = row2
                break
            row = row2

        snap = {
            "bid_depth": bid_depth,
            "ask_depth": ask_depth,
            "bid_notional": bid_notional,
            "ask_notional": ask_notional,
        }
        return ts, snap


# =========================
# FEATURES
# =========================

def book_features_missing():
    bd = np.zeros(5, dtype=np.float32)
    ad = np.zeros(5, dtype=np.float32)
    bn = np.zeros(5, dtype=np.float32)
    an = np.zeros(5, dtype=np.float32)
    depth_imb = np.zeros(5, dtype=np.float32)
    not_imb = np.zeros(5, dtype=np.float32)
    extra = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0.0, 0.0], dtype=np.float32)
    return np.concatenate([bd, ad, bn, an, depth_imb, not_imb, extra]).astype(np.float32)


def book_features_from_snap(snap, bin_end_ms: int, last_book_ts: int):
    bd = snap["bid_depth"]
    ad = snap["ask_depth"]
    bn = snap["bid_notional"]
    an = snap["ask_notional"]

    depth_imb = (bd - ad) / (bd + ad + EPS)
    not_imb = (bn - an) / (bn + an + EPS)

    cum_bd = float(bd.sum())
    cum_ad = float(ad.sum())
    cum_bn = float(bn.sum())
    cum_an = float(an.sum())

    shape_bd = float(bd[0] / (bd[4] + EPS))
    shape_ad = float(ad[0] / (ad[4] + EPS))
    shape_bn = float(bn[0] / (bn[4] + EPS))
    shape_an = float(an[0] / (an[4] + EPS))

    age_ms = float(bin_end_ms - last_book_ts)

    extra = np.array(
        [cum_bd, cum_ad, cum_bn, cum_an, shape_bd, shape_ad, shape_bn, shape_an, age_ms, 1.0],
        dtype=np.float32
    )
    return np.concatenate([bd, ad, bn, an, depth_imb, not_imb, extra]).astype(np.float32)


def trade_features_from_bin(
    has_trades: bool,
    o: float, h: float, l: float, c: float,
    sum_qty: float, sum_quote: float, sum_pq: float,
    n: int, buy_qty: float, sell_qty: float,
    prev_close: float,
    sum_logret_sq_inbin: float
):
    if prev_close > 0 and c > 0:
        logret = float(math.log(c / prev_close))
    else:
        logret = 0.0

    vwap = float(sum_pq / sum_qty) if sum_qty > 0 else float(c)
    delta = float(buy_qty - sell_qty)
    ofi = float(delta / (buy_qty + sell_qty + EPS))
    rv = float(math.sqrt(max(0.0, sum_logret_sq_inbin)))

    feats = np.array([
        1.0 if has_trades else 0.0,
        float(n),
        float(sum_qty),
        float(sum_quote),
        float(vwap),
        float(o),
        float(h),
        float(l),
        float(c),
        float(logret),
        float(rv),
        float(buy_qty),
        float(sell_qty),
        float(delta),
        float(ofi),
    ], dtype=np.float32)
    return feats, logret


def make_feature_vector(trade_feats: np.ndarray, book_feats: np.ndarray):
    return np.concatenate([trade_feats, book_feats]).astype(np.float32)


def feature_dims():
    trade_d = 15
    book_d = len(book_features_missing())
    return trade_d, book_d, trade_d + book_d


# =========================
# SHARDED WRITER
# =========================
class ShardWriter:
    def __init__(self, out_dir: Path, scale_name: str, interval_ms: int, shard_rows: int):
        self.out_dir = out_dir
        self.scale_name = scale_name
        self.interval_ms = int(interval_ms)
        self.shard_rows = int(shard_rows)

        self.x_dir = out_dir / scale_name / "X"
        self.t_dir = out_dir / scale_name / "T"
        self.x_dir.mkdir(parents=True, exist_ok=True)
        self.t_dir.mkdir(parents=True, exist_ok=True)

        self.shards = []
        self.total_rows = 0
        self.shard_idx = 0
        self.d = None
        self.first_ts = None
        self.last_ts = None

        self._X = None
        self._T = None
        self._n = 0

    def _ensure_buf(self, d: int):
        if self._X is None:
            self._X = np.empty((self.shard_rows, d), dtype=np.float32)
            self._T = np.empty((self.shard_rows,), dtype=np.int64)
            self._n = 0

    def add(self, t_ms: int, x: np.ndarray):
        t_ms = int(t_ms)
        if self.d is None:
            self.d = int(x.shape[-1])
        self._ensure_buf(self.d)

        if self.first_ts is None:
            self.first_ts = t_ms
        self.last_ts = t_ms

        self._X[self._n] = x.astype(np.float32, copy=False)
        self._T[self._n] = t_ms
        self._n += 1

        if self._n >= self.shard_rows:
            self.flush()

    def flush(self):
        if self._n <= 0:
            return

        X = self._X[:self._n].copy()
        T = self._T[:self._n].copy()

        x_path = self.x_dir / f"X_{self.shard_idx:06d}.npy"
        t_path = self.t_dir / f"T_{self.shard_idx:06d}.npy"
        np.save(x_path, X)
        np.save(t_path, T)

        shard = {
            "x": str(x_path.as_posix()),
            "t": str(t_path.as_posix()),
            "rows": int(X.shape[0]),
            "start_ts": int(T[0]),
            "end_ts": int(T[-1]),
            "offset": int(self.total_rows),
        }
        self.shards.append(shard)
        self.total_rows += int(X.shape[0])
        self.shard_idx += 1
        self._n = 0

    def save_meta(self):
        self.flush()
        meta = {
            "scale": self.scale_name,
            "interval_ms": int(self.interval_ms),
            "d": self.d,
            "total_rows": self.total_rows,
            "first_ts": int(self.first_ts) if self.first_ts is not None else None,
            "last_ts": int(self.last_ts) if self.last_ts is not None else None,
            "shards": self.shards,
        }
        meta_path = self.out_dir / self.scale_name / "meta.json"
        with meta_path.open("w") as f:
            json.dump(meta, f, indent=2)


# =========================
# RUNNING MOMENTS
# =========================
class RunningMoments:
    def __init__(self, d: int):
        self.d = int(d)
        self.n = 0
        self.mean = np.zeros(self.d, dtype=np.float64)
        self.m2 = np.zeros(self.d, dtype=np.float64)

    def update(self, x: np.ndarray):
        x = x.astype(np.float64)
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.m2 += delta * delta2

    def finalize(self):
        if self.n < 2:
            var = np.ones(self.d, dtype=np.float64)
        else:
            var = self.m2 / (self.n - 1)
        std = np.sqrt(np.maximum(var, 1e-12))
        return self.mean.astype(np.float32), std.astype(np.float32)


# =========================
# COARSE AGGREGATOR
# =========================
class CoarseAgg:
    def __init__(self, book_d: int):
        self.book_d = int(book_d)
        self.prev_interval_close = 0.0
        self.reset_interval()

    def reset_interval(self):
        self.started = False
        self.o = 0.0
        self.h = -1e30
        self.l = 1e30
        self.c = 0.0
        self.sum_qty = 0.0
        self.sum_quote = 0.0
        self.sum_pq = 0.0
        self.n_trades = 0
        self.buy_qty = 0.0
        self.sell_qty = 0.0
        self.sum_logret_sq = 0.0

        self.book_count = 0
        self.book_sum = np.zeros(self.book_d, dtype=np.float64)
        self.book_last = np.zeros(self.book_d, dtype=np.float32)

    def update_from_base(self, base_trade, base_book_feats, base_logret):
        o = float(base_trade[5])
        h = float(base_trade[6])
        l = float(base_trade[7])
        c = float(base_trade[8])

        if not self.started:
            self.started = True
            self.o = o
            self.h = h
            self.l = l
            self.c = c
        else:
            self.h = max(self.h, h)
            self.l = min(self.l, l)
            self.c = c

        self.sum_qty += float(base_trade[2])
        self.sum_quote += float(base_trade[3])
        if float(base_trade[2]) > 0:
            self.sum_pq += float(base_trade[4]) * float(base_trade[2])

        self.n_trades += int(base_trade[1])
        self.buy_qty += float(base_trade[11])
        self.sell_qty += float(base_trade[12])
        self.sum_logret_sq += float(base_logret) * float(base_logret)

        bf = base_book_feats.astype(np.float32)
        self.book_last = bf
        self.book_sum += bf.astype(np.float64)
        self.book_count += 1

    def flush(self):
        if not self.started:
            self.reset_interval()
            return None

        c = float(self.c)
        prev = float(self.prev_interval_close) if self.prev_interval_close > 0 else c
        logret = float(math.log(c / prev)) if prev > 0 and c > 0 else 0.0

        vwap = float(self.sum_pq / self.sum_qty) if self.sum_qty > 0 else c
        delta = float(self.buy_qty - self.sell_qty)
        ofi = float(delta / (self.buy_qty + self.sell_qty + EPS))
        rv = float(math.sqrt(max(0.0, self.sum_logret_sq)))

        trade_feats = np.array([
            1.0 if self.n_trades > 0 else 0.0,
            float(self.n_trades),
            float(self.sum_qty),
            float(self.sum_quote),
            float(vwap),
            float(self.o),
            float(self.h),
            float(self.l),
            float(self.c),
            float(logret),
            float(rv),
            float(self.buy_qty),
            float(self.sell_qty),
            float(delta),
            float(ofi),
        ], dtype=np.float32)

        if self.book_count > 0:
            book_mean = (self.book_sum / float(self.book_count)).astype(np.float32)
            book_feats = np.concatenate([book_mean, self.book_last]).astype(np.float32)
        else:
            z = np.zeros(self.book_d, dtype=np.float32)
            book_feats = np.concatenate([z, z]).astype(np.float32)

        x = np.concatenate([trade_feats, book_feats]).astype(np.float32)
        self.prev_interval_close = c
        self.reset_interval()
        return x


# =========================
# PREPARE DATASET
# =========================
def prepare_dataset():
    seed_all(SEED)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    trade_files_wd = list_sorted_files(TRADES_DIR, "trades")
    book_files_wd = list_sorted_files(BOOK_DIR, "bookDepth")
    if not trade_files_wd or not book_files_wd:
        raise RuntimeError("Missing trades or bookDepth files")

    start_date = max(trade_files_wd[0][0], book_files_wd[0][0])
    end_date = min(trade_files_wd[-1][0], book_files_wd[-1][0])

    trade_files = filter_files_by_date(trade_files_wd, start_date, end_date)
    book_files = filter_files_by_date(book_files_wd, start_date, end_date)
    if not trade_files or not book_files:
        raise RuntimeError("No overlapping files after trimming by date")

    first_book_row = read_first_data_row_csv(book_files[0])
    last_book_row = read_last_data_row_csv(book_files[-1])
    first_trade_row = read_first_data_row_csv(trade_files[0])
    last_trade_row = read_last_data_row_csv(trade_files[-1])

    if first_book_row is None or last_book_row is None or first_trade_row is None or last_trade_row is None:
        raise RuntimeError("Could not read first or last rows")

    first_book_ts = book_ts_to_ms(first_book_row[0].strip())
    last_book_ts = book_ts_to_ms(last_book_row[0].strip())
    first_trade_ts = int(first_trade_row[4])
    last_trade_ts = int(last_trade_row[4])

    t_start = max(first_book_ts, first_trade_ts)
    t_end = min(last_book_ts, last_trade_ts)
    if t_end <= t_start:
        raise RuntimeError("No overlap after trimming")

    train_end_ms = utc_str_to_ms(TRAIN_END_UTC)
    val_end_ms = utc_str_to_ms(VAL_END_UTC)

    _, book_d, full_d = feature_dims()

    writers = {
        "250ms": ShardWriter(OUT_ROOT, "250ms", SCALES_MS["250ms"], SHARD_ROWS["250ms"]),
        "20s": ShardWriter(OUT_ROOT, "20s", SCALES_MS["20s"], SHARD_ROWS["20s"]),
        "5m": ShardWriter(OUT_ROOT, "5m", SCALES_MS["5m"], SHARD_ROWS["5m"]),
        "1h": ShardWriter(OUT_ROOT, "1h", SCALES_MS["1h"], SHARD_ROWS["1h"]),
    }

    moments = {
        "250ms": RunningMoments(full_d),
        "20s": RunningMoments(15 + 2 * book_d),
        "5m": RunningMoments(15 + 2 * book_d),
        "1h": RunningMoments(15 + 2 * book_d),
    }

    trades = TradeStream(trade_files, t_start, t_end)
    books = BookStream(book_files, t_start, t_end)

    t0 = ceil_to_grid_ms(t_start, SCALES_MS["1h"])
    t_end_floor = floor_to_grid_ms(t_end, BASE_MS)
    if t0 + BASE_MS > t_end_floor:
        raise RuntimeError("Overlap too short after alignment")

    next_book = books.next_snapshot()
    last_book = None
    last_book_ts2 = 0
    while next_book is not None and next_book[0] <= t0:
        last_book_ts2, last_book = next_book
        next_book = books.next_snapshot()

    prev_bin_close = 0.0
    cur_trade = trades.pop()
    while cur_trade is not None and cur_trade[0] <= t0:
        prev_bin_close = float(cur_trade[1])
        cur_trade = trades.pop()

    if prev_bin_close <= 0.0:
        raise RuntimeError("No trade at or before aligned start t0")

    agg20 = CoarseAgg(book_d)
    agg5m = CoarseAgg(book_d)
    agg1h = CoarseAgg(book_d)

    next_end_20 = t0 + SCALES_MS["20s"]
    next_end_5m = t0 + SCALES_MS["5m"]
    next_end_1h = t0 + SCALES_MS["1h"]

    trade_sum_qty = 0.0
    trade_sum_quote = 0.0
    trade_sum_pq = 0.0
    trade_n = 0
    trade_buy_qty = 0.0
    trade_sell_qty = 0.0
    trade_o = 0.0
    trade_h = -1e30
    trade_l = 1e30
    trade_c = 0.0
    trade_has = False
    trade_prev_price_inbin = 0.0
    trade_sum_logret_sq_inbin = 0.0

    def reset_trade_bin():
        nonlocal trade_sum_qty, trade_sum_quote, trade_sum_pq, trade_n
        nonlocal trade_buy_qty, trade_sell_qty, trade_o, trade_h, trade_l, trade_c, trade_has
        nonlocal trade_prev_price_inbin, trade_sum_logret_sq_inbin
        trade_sum_qty = 0.0
        trade_sum_quote = 0.0
        trade_sum_pq = 0.0
        trade_n = 0
        trade_buy_qty = 0.0
        trade_sell_qty = 0.0
        trade_o = 0.0
        trade_h = -1e30
        trade_l = 1e30
        trade_c = 0.0
        trade_has = False
        trade_prev_price_inbin = 0.0
        trade_sum_logret_sq_inbin = 0.0

    def update_trade_bin(price, qty, quote_qty, is_buyer_maker):
        nonlocal trade_sum_qty, trade_sum_quote, trade_sum_pq, trade_n
        nonlocal trade_buy_qty, trade_sell_qty, trade_o, trade_h, trade_l, trade_c, trade_has
        nonlocal trade_prev_price_inbin, trade_sum_logret_sq_inbin

        trade_sum_qty += float(qty)
        trade_sum_quote += float(quote_qty)
        trade_sum_pq += float(price) * float(qty)
        trade_n += 1

        if not is_buyer_maker:
            trade_buy_qty += float(qty)
        else:
            trade_sell_qty += float(qty)

        if not trade_has:
            trade_has = True
            trade_o = float(price)
            trade_h = float(price)
            trade_l = float(price)
            trade_prev_price_inbin = float(price)
        else:
            trade_h = max(trade_h, float(price))
            trade_l = min(trade_l, float(price))
            pr = float(trade_prev_price_inbin)
            if pr > 0 and float(price) > 0:
                lr = math.log(float(price) / pr)
                trade_sum_logret_sq_inbin += float(lr) * float(lr)
            trade_prev_price_inbin = float(price)

        trade_c = float(price)

    total_bins = int((t_end_floor - (t0 + BASE_MS)) // BASE_MS) + 1
    pbar = tqdm(total=max(1, total_bins), desc="prepare", dynamic_ncols=True)

    bin_end = t0 + BASE_MS
    while bin_end <= t_end_floor:
        while next_book is not None and next_book[0] <= bin_end:
            last_book_ts2, last_book = next_book
            next_book = books.next_snapshot()

        while cur_trade is not None and cur_trade[0] < bin_end:
            _, price, qty, quote_qty, is_buyer_maker = cur_trade
            update_trade_bin(price, qty, quote_qty, is_buyer_maker)
            cur_trade = trades.pop()

        if trade_has:
            o = trade_o
            h = trade_h
            l = trade_l
            c = trade_c
        else:
            o = prev_bin_close
            h = prev_bin_close
            l = prev_bin_close
            c = prev_bin_close

        trade_feats, base_logret = trade_features_from_bin(
            trade_has,
            o, h, l, c,
            trade_sum_qty, trade_sum_quote, trade_sum_pq,
            trade_n, trade_buy_qty, trade_sell_qty,
            prev_bin_close,
            trade_sum_logret_sq_inbin
        )

        if last_book is None:
            book_feats = book_features_missing()
        else:
            book_feats = book_features_from_snap(last_book, bin_end, last_book_ts2)

        x250 = make_feature_vector(trade_feats, book_feats)
        writers["250ms"].add(bin_end, x250)

        agg20.update_from_base(trade_feats, book_feats, base_logret)
        agg5m.update_from_base(trade_feats, book_feats, base_logret)
        agg1h.update_from_base(trade_feats, book_feats, base_logret)

        if bin_end == next_end_20:
            x20 = agg20.flush()
            if x20 is not None:
                writers["20s"].add(bin_end, x20)
                if bin_end < train_end_ms:
                    moments["20s"].update(x20)
            next_end_20 += SCALES_MS["20s"]

        if bin_end == next_end_5m:
            x5m = agg5m.flush()
            if x5m is not None:
                writers["5m"].add(bin_end, x5m)
                if bin_end < train_end_ms:
                    moments["5m"].update(x5m)
            next_end_5m += SCALES_MS["5m"]

        if bin_end == next_end_1h:
            x1h = agg1h.flush()
            if x1h is not None:
                writers["1h"].add(bin_end, x1h)
                if bin_end < train_end_ms:
                    moments["1h"].update(x1h)
            next_end_1h += SCALES_MS["1h"]

        if bin_end < train_end_ms:
            moments["250ms"].update(x250)

        prev_bin_close = float(c)
        reset_trade_bin()
        bin_end += BASE_MS
        pbar.update(1)

    pbar.close()

    for k in writers:
        writers[k].save_meta()

    norm = {}
    for k, rm in moments.items():
        mean, std = rm.finalize()
        norm[k] = {"mean": mean.tolist(), "std": std.tolist()}

    meta = {
        "symbol": SYMBOL,
        "base_ms": BASE_MS,
        "scales_ms": SCALES_MS,
        "seq_len": SEQ_LEN,
        "train_end_ms": int(train_end_ms),
        "val_end_ms": int(val_end_ms),
        "norm": norm,
        "fee_open": FEE_OPEN,
        "fee_close": FEE_CLOSE,
        "leverage": LEVERAGE,
        "loss_threshold_frac": LOSS_THRESHOLD_FRAC,
        "t0_aligned_start_ms": int(t0),
        "t_end_floor_ms": int(t_end_floor),
    }
    with (OUT_ROOT / "dataset_meta.json").open("w") as f:
        json.dump(meta, f, indent=2)

    print("done prepare:", OUT_ROOT.as_posix())