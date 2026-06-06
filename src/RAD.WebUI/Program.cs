using System.Collections.Specialized;
using System.Diagnostics;
using System.Net;
using System.Text;
using System.Web;
using RAD.Detector;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Formats.Png;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

namespace RAD.WebUI;

class Program
{
    static RadInference? _inference;
    static bool _bankLoaded;
    static string _modelPath = "dinov3_multilayer.onnx";
    static string _device = "CPU";
    static float _threshold = 0.12f;
    static int _kImage = 10;
    static string? _category;
    static readonly StringBuilder _logBuf = new();
    static string LogText => _logBuf.ToString();
    static double _preMs, _boneMs, _postMs, _vizMs, _totalMs;
    static readonly string _baseDir = AppDomain.CurrentDomain.BaseDirectory;
    static string ImagesDir => Path.Combine(_baseDir, "images");

    static void Main(string[] args)
    {
        var port = args.Length > 0 && int.TryParse(args[0], out var p) ? p : 8080;

        Directory.CreateDirectory(ImagesDir);
        TryLoadModel();

        var listener = new HttpListener();
        listener.Prefixes.Add($"http://localhost:{port}/");
        listener.Prefixes.Add($"http://127.0.0.1:{port}/");
        listener.Start();
        Console.WriteLine($"RAD WebUI at http://localhost:{port}/");

        while (true)
        {
            var ctx = listener.GetContext();
            ThreadPool.QueueUserWorkItem(_ => HandleRequest(ctx));
        }
    }

    static void Log(string msg)
    {
        var ts = DateTime.Now.ToString("HH:mm:ss");
        var line = $"[{ts}] {msg}";
        lock (_logBuf) { _logBuf.Insert(0, line + "\n"); if (_logBuf.Length > 10000) _logBuf.Length = 10000; }
        Console.WriteLine(line);
    }

    static string TimingText => _bankLoaded
        ? $"Pre: {_preMs:F1}ms  Bone: {_boneMs:F1}ms  Post: {_postMs:F1}ms  Viz: {_vizMs:F1}ms  Total: {_totalMs:F1}ms"
        : "Pre: --ms  Bone: --ms  Post: --ms  Viz: --ms  Total: --ms";

    static void TryLoadModel()
    {
        var path = Path.Combine(_baseDir, _modelPath);
        if (File.Exists(path))
        {
            _inference = new RadInference(path, _device, _kImage);
            Log($"Model loaded ({_inference.ActiveDevice}): {Path.GetFileName(path)}");
        }
        else Log($"Model not found: {path}");
    }

    // ====== Router ======
    static void HandleRequest(HttpListenerContext ctx)
    {
        try
        {
            var req = ctx.Request;
            var resp = ctx.Response;
            resp.ContentType = "text/html; charset=utf-8";
            var path = req.Url!.AbsolutePath.TrimStart('/');
            var qs = HttpUtility.ParseQueryString(req.Url.Query);

            switch (path)
            {
                case "": Serve(resp, MainPage()); break;
                case "set-params": SetParams(qs, resp); break;
                case "reload-model": ReloadModel(qs, resp); break;
                case "create-category": CreateCategory(qs, resp); break;
                case "build-bank": BuildBank(qs, resp); break;
                case "build-bank-stream": BuildBankStream(qs, resp); break;
                case "detect": Detect(qs, resp); break;
                case "upload": Upload(req, resp); break;
                case "delete-image": DeleteImage(qs, resp); break;
                case "preview": ServePreview(ctx); break;
                case "favicon.ico": resp.StatusCode = 404; resp.Close(); break;
                default: Serve(resp, HtmlTemplates.ErrorPage("Not Found")); break;
            }
        }
        catch (Exception ex) { try { Serve(ctx.Response, HtmlTemplates.ErrorPage(ex.Message)); } catch { } }
    }

    // ====== Pages ======
    static string MainPage()
    {
        var catOpts = "<option value=\"\">-- Select category --</option>";
        foreach (var d in GetCategories())
            catOpts += $"<option value=\"{H(d)}\"{(d == _category ? " selected" : "")}>{d}</option>";

        var testOpts = "<option value=\"\">-- Select image --</option>";
        if (_category != null)
        {
            var testDir = Path.Combine(ImagesDir, _category, "test");
            if (Directory.Exists(testDir))
                foreach (var f in ImageFiles(testDir))
                    testOpts += $"<option value=\"{H(f)}\">{Path.GetFileName(f)}</option>";
        }

        return HtmlTemplates.MainPage(_modelPath, _device, _kImage, _threshold,
            _bankLoaded, _category, LogText, TimingText, catOpts, testOpts);
    }

    static void ServePreview(HttpListenerContext ctx)
    {
        var path = HttpUtility.ParseQueryString(ctx.Request.Url!.Query)["path"];
        if (path == null || !File.Exists(path)) { ctx.Response.StatusCode = 404; ctx.Response.Close(); return; }
        using var img = Image.Load(path);
        var scale = Math.Min(200f / Math.Max(img.Width, img.Height), 1f);
        img.Mutate(x => x.Resize((int)(img.Width * scale), (int)(img.Height * scale)));
        ctx.Response.ContentType = "image/jpeg";
        using var ms = new MemoryStream();
        img.SaveAsJpeg(ms);
        var bytes = ms.ToArray();
        ctx.Response.ContentLength64 = bytes.Length;
        ctx.Response.OutputStream.Write(bytes);
        ctx.Response.OutputStream.Close();
    }

    // ====== Actions ======
    static void SetParams(NameValueCollection qs, HttpListenerResponse resp)
    {
        if (qs["model"] is string m and { Length: > 0 }) _modelPath = m;
        if (qs["device"] is string d) { _device = d; _inference?.SwitchDevice(d); }
        if (float.TryParse(qs["threshold"], out var t)) _threshold = t;
        if (int.TryParse(qs["kimage"], out var k) && k > 0) { _kImage = k; if (_inference != null) _inference.KImage = k; }
        if (qs["category"] is string c) _category = c.Length > 0 ? c : null;
        Serve(resp, HtmlTemplates.RedirectPage(qs["back"] ?? "/"));
    }

    static void ReloadModel(NameValueCollection qs, HttpListenerResponse resp)
    {
        if (qs["model"] is string m) _modelPath = m;
        if (qs["device"] is string d) _device = d;
        _inference?.Dispose();
        TryLoadModel();
        if (_inference != null)
        {
            // Reload bank if category is set
            if (_category != null)
            {
                var bankDir = Path.Combine(ImagesDir, _category, ".bankdata");
                if (Directory.Exists(bankDir)) { _inference.LoadBank(bankDir); _bankLoaded = true; }
            }
        }
        ServeText(resp, _inference != null ? $"Loaded ({_inference.ActiveDevice})" : "FAILED");
    }

    static void CreateCategory(NameValueCollection qs, HttpListenerResponse resp)
    {
        var name = (qs["name"] ?? "").Trim();
        if (string.IsNullOrEmpty(name) || name.IndexOfAny(Path.GetInvalidFileNameChars()) >= 0)
        { Serve(resp, HtmlTemplates.ErrorPage("Invalid name")); return; }
        var d = Path.Combine(ImagesDir, name);
        Directory.CreateDirectory(d);
        Directory.CreateDirectory(Path.Combine(d, "bank"));
        Directory.CreateDirectory(Path.Combine(d, "test"));
        _category = name;
        _bankLoaded = false;
        Log($"Created category: {name}");
        Serve(resp, HtmlTemplates.RedirectPage("/?category=" + HttpUtility.UrlEncode(name)));
    }

    static void BuildBank(NameValueCollection qs, HttpListenerResponse resp)
    {
        // Just return the page that will open SSE stream
        var cat = qs["category"] ?? _category ?? "";
        Serve(resp, HtmlTemplates.BuildBankPage(cat));
    }

    static void BuildBankStream(NameValueCollection qs, HttpListenerResponse resp)
    {
        var cat = qs["category"] ?? _category ?? "";
        if (_inference == null) { Serve(resp, HtmlTemplates.ErrorPage("Model not loaded")); return; }
        if (string.IsNullOrEmpty(cat)) { Serve(resp, HtmlTemplates.ErrorPage("No category selected")); return; }

        var bankDir = Path.Combine(ImagesDir, cat, "bank");
        Directory.CreateDirectory(bankDir);
        var bankOut = Path.Combine(ImagesDir, cat, ".bankdata");

        resp.ContentType = "text/event-stream";
        resp.Headers.Add("Cache-Control", "no-cache");
        resp.Headers.Add("Connection", "keep-alive");
        var writer = new StreamWriter(resp.OutputStream, Encoding.UTF8) { AutoFlush = true };

        void Sse(string evt, string data)
        {
            writer.WriteLine($"event: {evt}");
            writer.WriteLine($"data: {data}");
            writer.WriteLine();
        }

        Sse("log", "Building bank...");
        var sw = Stopwatch.StartNew();
        try
        {
            var (p, t) = Task.Run(() => _inference.BuildBank(bankDir, bankOut, msg =>
            {
                Sse("log", msg);
            })).Result;
            _bankLoaded = true; _category = cat;
            Sse("log", $"Bank built: {p}/{t} images, {sw.Elapsed.TotalSeconds:F1}s");
            Sse("done", $"/?category={HttpUtility.UrlEncode(cat)}");
        }
        catch (Exception ex)
        {
            Sse("log", $"Build failed: {ex.Message}");
            _bankLoaded = false;
            Sse("done", $"/?category={HttpUtility.UrlEncode(cat)}");
        }
        writer.Close();
    }

    static void Detect(NameValueCollection qs, HttpListenerResponse resp)
    {
        var cat = qs["category"] ?? _category ?? "";
        var file = qs["file"] ?? "";

        if (_inference == null) { Serve(resp, HtmlTemplates.ErrorPage("Model not loaded")); return; }
        if (string.IsNullOrEmpty(cat)) { Serve(resp, HtmlTemplates.ErrorPage("No category selected")); return; }
        if (!File.Exists(file)) { Serve(resp, HtmlTemplates.ErrorPage("File not found")); return; }

        var bankData = Path.Combine(ImagesDir, cat, ".bankdata");
        if (!_bankLoaded || _category != cat)
        {
            if (Directory.Exists(bankData)) { _inference.LoadBank(bankData); _bankLoaded = true; _category = cat; }
            else { Serve(resp, HtmlTemplates.ErrorPage("Bank not built for this category. Build it first.")); return; }
        }

        _inference.KImage = _kImage;
        var totalSw = Stopwatch.StartNew();
        var (amap, score) = _inference.Detect(file, _threshold, out _preMs, out _boneMs, out _postMs);
        var vizSw = Stopwatch.StartNew();

        using var orig = ImagePreprocessor.LoadForDisplay(file);
        using var heat = VisualizationHelper.CreateHeatmap(amap, orig.Width, orig.Height);
        using var over = VisualizationHelper.CreateOverlay(orig, heat, 0.35f);
        using var mask = VisualizationHelper.CreateBinaryMask(amap, _threshold, orig.Width, orig.Height);
        using var origRgba = orig.CloneAs<Rgba32>();
        var ds = 256;
        using var oD = VisualizationHelper.ResizeForDisplay(origRgba, ds);
        using var hD = VisualizationHelper.ResizeForDisplay(heat, ds);
        using var vD = VisualizationHelper.ResizeForDisplay(over, ds);
        using var mD = VisualizationHelper.ResizeForDisplay(mask, ds);
        _vizMs = vizSw.Elapsed.TotalMilliseconds; _totalMs = totalSw.Elapsed.TotalMilliseconds;

        Log($"Detection complete. Score: {score:F4}");
        var json = $"{{\"orig\":\"data:image/png;base64,{ImgToB64(oD)}\",\"heat\":\"data:image/png;base64,{ImgToB64(hD)}\",\"over\":\"data:image/png;base64,{ImgToB64(vD)}\",\"mask\":\"data:image/png;base64,{ImgToB64(mD)}\",\"score\":{score:F4},\"timing\":\"{TimingText.Replace("\"", "\\\"")}\"}}";
        resp.ContentType = "application/json; charset=utf-8";
        ServeTextRaw(resp, json);
    }

    static void Upload(HttpListenerRequest req, HttpListenerResponse resp)
    {
        var (fields, fileCount, lastErr) = ParseMultipart(req);
        var cat = fields.Get("cat") ?? fields.Get("category") ?? _category ?? "";
        var type = fields.Get("tp") ?? fields.Get("type") ?? "bank";
        if (string.IsNullOrEmpty(cat)) { Serve(resp, HtmlTemplates.ErrorPage("No category")); return; }

        var folder = Path.Combine(ImagesDir, cat, type);
        Directory.CreateDirectory(folder);
        Log($"Uploaded {fileCount} image(s) to {cat}/{type}" + (lastErr != null ? $" (1 error: {lastErr})" : ""));

        var back = $"/?category=" + HttpUtility.UrlEncode(cat);
        Serve(resp, HtmlTemplates.RedirectPage(back));
    }

    static void DeleteImage(NameValueCollection qs, HttpListenerResponse resp)
    {
        var file = qs["path"] ?? "";
        if (File.Exists(file)) { File.Delete(file); Log($"Deleted: {file}"); }
        Serve(resp, HtmlTemplates.RedirectPage(qs["back"] ?? "/"));
    }

    // ====== Helpers ======
    static string[] GetCategories()
    {
        if (!Directory.Exists(ImagesDir)) return [];
        return Directory.EnumerateDirectories(ImagesDir)
            .Select(Path.GetFileName).Where(n => n != null && !n.StartsWith(".")).OrderBy(n => n).ToArray()!;
    }

    static string[] ImageFiles(string dir) =>
        Directory.Exists(dir)
            ? Directory.EnumerateFiles(dir).Where(f =>
                f.EndsWith(".jpg", StringComparison.OrdinalIgnoreCase) ||
                f.EndsWith(".jpeg", StringComparison.OrdinalIgnoreCase) ||
                f.EndsWith(".png", StringComparison.OrdinalIgnoreCase) ||
                f.EndsWith(".bmp", StringComparison.OrdinalIgnoreCase)).OrderBy(f => f).ToArray()
            : [];

    static (NameValueCollection fields, int fileCount, string? error) ParseMultipart(HttpListenerRequest req)
    {
        var fields = System.Web.HttpUtility.ParseQueryString("");
        var ct = req.ContentType ?? "";
        if (!ct.Contains("multipart/form-data")) return (fields, 0, null);
        var boundary = Encoding.ASCII.GetBytes("--" + ct.Split("boundary=").Last().Trim('"', ' '));
        var buf = new byte[req.ContentLength64];
        var read = 0;
        while (read < buf.Length) read += req.InputStream.Read(buf, read, buf.Length - read);
        int count = 0, pos = 0;
        string? lastErr = null;
        while (pos < buf.Length)
        {
            var nb = IndexOf(buf, boundary, pos); if (nb < 0) break;
            pos = nb + boundary.Length;
            if (pos + 1 < buf.Length && buf[pos] == '\r' && buf[pos + 1] == '\n') pos += 2;
            var hdrEnd = IndexOf(buf, "\r\n\r\n"u8, pos); if (hdrEnd < 0) break;
            var hdr = Encoding.ASCII.GetString(buf, pos, hdrEnd - pos);
            var fnIdx = hdr.IndexOf("filename=\"", StringComparison.Ordinal);
            var dataStart = hdrEnd + 4;
            var nextB = IndexOf(buf, boundary, dataStart);
            var dataEnd = nextB >= 0 ? nextB - 2 : buf.Length;

            if (fnIdx >= 0)
            {
                // This part contains a file
                var ff = fnIdx + 10; var ffEnd = hdr.IndexOf('"', ff);
                if (ffEnd > ff)
                {
                    var fname = hdr[ff..ffEnd];
                    // Determine folder: find "cat" form field from previous parts
                    var catVal = fields["cat"]; var tpVal = fields["tp"] ?? "bank";
                    if (!string.IsNullOrEmpty(catVal))
                    {
                        var folder = Path.Combine(ImagesDir, catVal, tpVal);
                        Directory.CreateDirectory(folder);
                        if (dataEnd > dataStart)
                        {
                            try
                            {
                                var fpath = Path.Combine(folder, Path.GetFileName(fname));
                                File.WriteAllBytes(fpath, buf[dataStart..dataEnd]);
                                count++;
                            }
                            catch (Exception ex) { lastErr = ex.Message; }
                        }
                    }
                }
            }
            else
            {
                // This part is a form field
                if (dataEnd > dataStart)
                {
                    var text = Encoding.UTF8.GetString(buf, dataStart, dataEnd - dataStart);
                    var nameIdx = hdr.IndexOf("name=\"", StringComparison.Ordinal);
                    if (nameIdx >= 0)
                    {
                        var nm = nameIdx + 6; var nmEnd = hdr.IndexOf('"', nm);
                        if (nmEnd > nm)
                        {
                            var fieldName = hdr[nm..nmEnd];
                            fields[fieldName] = text;
                        }
                    }
                }
            }
            pos = Math.Max(dataStart, dataEnd);
        }
        return (fields, count, lastErr);
    }

    static int IndexOf(byte[] s, byte[] p, int start)
    { for (int i = start; i <= s.Length - p.Length; i++) { bool m = true; for (int j = 0; j < p.Length; j++) if (s[i + j] != p[j]) { m = false; break; } if (m) return i; } return -1; }
    static int IndexOf(byte[] s, ReadOnlySpan<byte> p, int start)
    { for (int i = start; i <= s.Length - p.Length; i++) { bool m = true; for (int j = 0; j < p.Length; j++) if (s[i + j] != p[j]) { m = false; break; } if (m) return i; } return -1; }

    static string ImgToB64(Image<Rgba32> img) { using var ms = new MemoryStream(); img.Save(ms, new PngEncoder()); return Convert.ToBase64String(ms.ToArray()); }
    static void Serve(HttpListenerResponse resp, string html) { var b = Encoding.UTF8.GetBytes(html); resp.ContentLength64 = b.Length; resp.OutputStream.Write(b); resp.OutputStream.Close(); }
    static void ServeText(HttpListenerResponse resp, string t) { var b = Encoding.UTF8.GetBytes(t); resp.ContentType = "text/plain; charset=utf-8"; resp.ContentLength64 = b.Length; resp.OutputStream.Write(b); resp.OutputStream.Close(); }
    static void ServeTextRaw(HttpListenerResponse resp, string t) { var b = Encoding.UTF8.GetBytes(t); resp.ContentLength64 = b.Length; resp.OutputStream.Write(b); resp.OutputStream.Close(); }
    static string H(string s) => WebUtility.HtmlEncode(s);
}
