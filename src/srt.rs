use crate::transcribe::Segment;

/// Format milidetik menjadi "HH:MM:SS,mmm" sesuai standar SRT.
pub fn format_timestamp(ms: i64) -> String {
    let ms = ms.max(0);
    let h = ms / 3_600_000;
    let m = (ms % 3_600_000) / 60_000;
    let s = (ms % 60_000) / 1000;
    let milli = ms % 1000;
    format!("{h:02}:{m:02}:{s:02},{milli:03}")
}

/// Bungkus teks per kata agar tiap baris tidak melebihi `max` karakter.
fn wrap(text: &str, max: usize) -> String {
    let mut lines: Vec<String> = Vec::new();
    let mut cur = String::new();
    for w in text.split_whitespace() {
        if !cur.is_empty() && cur.chars().count() + 1 + w.chars().count() > max {
            lines.push(std::mem::take(&mut cur));
        }
        if !cur.is_empty() {
            cur.push(' ');
        }
        cur.push_str(w);
    }
    if !cur.is_empty() {
        lines.push(cur);
    }
    lines.join("\n")
}

pub fn to_srt(segments: &[Segment], max_line_chars: usize) -> String {
    let mut out = String::new();
    for (i, seg) in segments.iter().enumerate() {
        out.push_str(&format!(
            "{}\n{} --> {}\n{}\n\n",
            i + 1,
            format_timestamp(seg.start_ms),
            format_timestamp(seg.end_ms),
            wrap(&seg.text, max_line_chars)
        ));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn timestamp_format() {
        assert_eq!(format_timestamp(0), "00:00:00,000");
        assert_eq!(format_timestamp(3_723_456), "01:02:03,456");
        assert_eq!(format_timestamp(-5), "00:00:00,000");
    }

    #[test]
    fn wrap_lines() {
        let t = wrap("satu dua tiga empat lima enam", 12);
        assert_eq!(t, "satu dua\ntiga empat\nlima enam");
    }

    #[test]
    fn srt_block() {
        let segs = vec![Segment {
            start_ms: 1000,
            end_ms: 2500,
            text: "Halo dunia".into(),
            ..Default::default()
        }];
        assert_eq!(
            to_srt(&segs, 42),
            "1\n00:00:01,000 --> 00:00:02,500\nHalo dunia\n\n"
        );
    }
}
