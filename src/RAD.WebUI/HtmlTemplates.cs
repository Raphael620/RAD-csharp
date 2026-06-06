namespace RAD.WebUI;

static class HtmlTemplates
{
    static string H(string s) => System.Net.WebUtility.HtmlEncode(s);

    public static string Css => @"
*{margin:0;padding:0;box-sizing:border-box}
body{font:14px -apple-system,BlinkMacSystemFont,sans-serif;background:#f5f5f5;color:#222;padding:16px;max-width:1200px;margin:0 auto}
h1{font-size:20px;margin-bottom:14px}h3{font-size:15px;margin-bottom:6px}
.row{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:8px;align-items:center}
label{font-weight:600;min-width:55px;font-size:13px}
input,select,button{padding:5px 8px;border:1px solid #ccc;border-radius:4px;font-size:13px}
input[type=text],input[type=number],input[type=file]{flex:1;min-width:100px}
button{background:#4a90d9;color:#fff;border:none;cursor:pointer;white-space:nowrap}
button:hover{background:#357abd}
button.small{font-size:11px;padding:2px 6px}
textarea{width:100%;height:90px;font-family:monospace;font-size:11px;padding:5px;border:1px solid #ccc;border-radius:4px;resize:vertical}
.card{background:#fff;border-radius:6px;padding:12px;margin-bottom:12px;box-shadow:0 1px 2px rgba(0,0,0,.08)}
.imgs{display:flex;gap:8px;flex-wrap:wrap}
.imgbox{text-align:center}
.imgbox img{width:260px;height:260px;object-fit:contain;border:1px solid #ddd;border-radius:4px;background:#fafafa}
.imgbox p{font-size:11px;margin-top:3px;color:#666}
.status{padding:4px 8px;border-radius:4px;font-size:12px;font-weight:600;display:inline-block}
.status.ok{background:#d4edda;color:#155724}.status.err{background:#f8d7da;color:#721c24}
.timing{font-family:monospace;font-size:11px;color:#555;margin-top:4px}
hr{margin:10px 0}
.thumb{display:inline-block;margin:4px;text-align:center;position:relative}
.thumb img{width:120px;height:120px;object-fit:contain;border:1px solid #ddd;border-radius:4px}
.thumb p{font-size:10px;margin:2px 0;max-width:120px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
".Trim();

    public static string Page(string title, string body, string? script = null) =>
        "<!DOCTYPE html><html lang=\"en\"><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">" +
        "<title>" + H(title) + " - RAD</title><style>" + Css + "</style></head><body>" +
        "<h1>" + H(title) + "</h1>" + body +
        (script != null ? "<script>" + script + "</script>" : "") + "</body></html>";

    public static string MainPage(string modelPath, string device, int kImage, float threshold,
        bool bankReady, string? category, string logText, string timingText,
        string categoryOptions, string testFileOptions)
    {
        var bankStatus = bankReady
            ? "<span class=\"status ok\">Bank Ready</span>"
            : "<span class=\"status err\">No Bank</span>";

        var catName = H(category ?? "");
        var existingUpload = category != null;
        var existingDetect = category != null && bankReady;

        return Page("RAD Anomaly Detection",
"<div class=\"card\">" +
"<h3>Model</h3>" +
"<div class=\"row\"><label>Path:</label><input id=\"model\" value=\"" + H(modelPath) + "\" style=\"flex:3\">" +
"<button onclick=\"reloadModel()\">Reload</button></div>" +
"<div class=\"row\"><label>Device:</label><select id=\"device\" onchange=\"setParams()\"><option" + (device == "CPU" ? " selected" : "") + ">CPU</option><option" + (device != "CPU" ? " selected" : "") + ">GPU</option></select>" +
"<label>kImage:</label><input id=\"kimage\" type=\"number\" min=\"1\" value=\"" + kImage + "\" onchange=\"setParams()\" style=\"width:65px\">" +
"<label>Threshold:</label><input id=\"threshold\" type=\"number\" step=\"0.01\" value=\"" + threshold.ToString("F2") + "\" onchange=\"setParams()\" style=\"width:75px\"></div>" +
"</div>" +

"<div class=\"card\">" +
"<h3>Category</h3>" +
"<div class=\"row\"><label>Select:</label><select id=\"category\" onchange=\"onCategory(this.value)\" style=\"flex:1\">" + categoryOptions + "</select>" +
"<button onclick=\"createCategory()\">+ New</button>" +
"<input id=\"newname\" placeholder=\"category name\" style=\"width:140px\" onkeydown=\"if(event.key==='Enter')createCategory()\"></div>" +
"<div class=\"row\">" +
"<form action=\"upload\" method=\"post\" enctype=\"multipart/form-data\" style=\"display:inline;flex:1\">" +
"<input type=\"hidden\" name=\"cat\" value=\"" + catName + "\">" +
"<input type=\"hidden\" name=\"tp\" value=\"bank\">" +
"<input type=\"file\" name=\"files\" multiple accept=\"image/*\" style=\"width:180px\">" +
"<button type=\"submit\" class=\"small\">Upload Bank</button></form>" +
"<form action=\"upload\" method=\"post\" enctype=\"multipart/form-data\" style=\"display:inline;flex:1\">" +
"<input type=\"hidden\" name=\"cat\" value=\"" + catName + "\">" +
"<input type=\"hidden\" name=\"tp\" value=\"test\">" +
"<input type=\"file\" name=\"files\" multiple accept=\"image/*\" style=\"width:180px\">" +
"<button type=\"submit\" class=\"small\">Upload Test</button></form></div>" +
"<div class=\"row\">" + bankStatus +
"<button onclick=\"buildBank()\" style=\"margin-left:8px\">Build Bank</button>" +
"<label style=\"margin-left:12px\">Test image:</label><select id=\"testFile\" style=\"flex:1\">" + testFileOptions + "</select>" +
"<button onclick=\"detect()\">Detect</button></div>" +
"<div class=\"timing\">" + H(timingText) + "</div>" +
"</div>" +

"<div class=\"card\"><h3>Log</h3><textarea readonly>" + H(logText) + "</textarea></div>" +

"<div class=\"card\">" +
"<div class=\"imgs\">" +
"<div class=\"imgbox\"><img id=\"imgOrig\" src=\"\"><p>Original</p></div>" +
"<div class=\"imgbox\"><img id=\"imgHeat\" src=\"\"><p>Anomaly Heatmap</p></div>" +
"<div class=\"imgbox\"><img id=\"imgOver\" src=\"\"><p>Overlay</p></div>" +
"<div class=\"imgbox\"><img id=\"imgMask\" src=\"\"><p>Defect Mask</p></div>" +
"</div></div>",

@"function setParams(){
 var m=document.getElementById('model').value;
 var d=document.getElementById('device').value;
 var t=document.getElementById('threshold').value;
 var k=document.getElementById('kimage').value;
 var c=document.getElementById('category').value;
 fetch('set-params?model='+encodeURIComponent(m)+'&device='+encodeURIComponent(d)+'&threshold='+t+'&kimage='+k+'&category='+encodeURIComponent(c)).then(()=>location.reload());
}
function reloadModel(){
 var m=document.getElementById('model').value;
 var d=document.getElementById('device').value;
 fetch('reload-model?model='+encodeURIComponent(m)+'&device='+encodeURIComponent(d)).then(r=>r.text()).then(t=>{alert(t);location.reload();});
}
function onCategory(c){ location.href='/?category='+encodeURIComponent(c); }
function createCategory(){
 var n=document.getElementById('newname').value.trim();
 if(!n){ alert('Enter a category name'); return; }
 location.href='create-category?name='+encodeURIComponent(n);
}
function buildBank(){
 var c=document.getElementById('category').value;
 if(!c){alert('Select a category first');return;}
 window.open('build-bank?category='+encodeURIComponent(c),'_self');
}
function detect(){
 var f=document.getElementById('testFile').value;
 var c=document.getElementById('category').value;
 if(!f){alert('Select a test image');return;}
 var btn=event.target;btn.disabled=true;btn.textContent='Detecting...';
 fetch('detect?category='+encodeURIComponent(c)+'&file='+encodeURIComponent(f))
  .then(r=>r.json()).then(data=>{
    document.getElementById('imgOrig').src=data.orig;
    document.getElementById('imgHeat').src=data.heat;
    document.getElementById('imgOver').src=data.over;
    document.getElementById('imgMask').src=data.mask;
    document.querySelector('.timing').textContent=data.timing;
    btn.disabled=false;btn.textContent='Detect';
  }).catch(e=>{alert(e);btn.disabled=false;btn.textContent='Detect';});
}");
    }

    public static string BuildBankPage(string category) =>
        "<!DOCTYPE html><html><head><meta charset=\"utf-8\"><title>Building Bank...</title><style>" + Css + "</style></head><body>" +
        "<div class=\"card\"><h3>Building Bank [" + H(category) + "]</h3>" +
        "<textarea id=\"log\" readonly style=\"width:100%;height:400px;font-family:monospace;font-size:12px\"></textarea>" +
        "<div id=\"status\" class=\"timing\">Starting...</div></div>" +
        "<script>" +
        "var src=new EventSource('build-bank-stream?category=" + System.Web.HttpUtility.UrlEncode(category) + "');" +
        "var ta=document.getElementById('log');var st=document.getElementById('status');" +
        "src.addEventListener('log',function(e){ta.value=e.data+'\\n'+ta.value;st.textContent='Processing...';});" +
        "src.addEventListener('done',function(e){st.textContent='Done!';src.close();setTimeout(function(){location=e.data;},500);});" +
        "src.onerror=function(){st.textContent='Connection error';};" +
        "</script></body></html>";

    public static string ResultPage(string fileName, float score, string origB64, string heatB64,
        string overB64, string maskB64, string timingText, string logText, string backUrl) =>
        Page("Detection Result",
"<div class=\"card\"><p><strong>File:</strong> " + H(fileName) + "</p>" +
"<p><strong>Score:</strong> " + score.ToString("F4") + "</p>" +
"<div class=\"timing\">" + H(timingText) + "</div>" +
"<div class=\"imgs\">" +
"<div class=\"imgbox\"><img src=\"data:image/png;base64," + origB64 + "\"><p>Original</p></div>" +
"<div class=\"imgbox\"><img src=\"data:image/png;base64," + heatB64 + "\"><p>Anomaly Heatmap</p></div>" +
"<div class=\"imgbox\"><img src=\"data:image/png;base64," + overB64 + "\"><p>Overlay</p></div>" +
"<div class=\"imgbox\"><img src=\"data:image/png;base64," + maskB64 + "\"><p>Defect Mask</p></div>" +
"</div></div>" +
"<div class=\"card\"><h3>Log</h3><textarea readonly>" + H(logText) + "</textarea></div>" +
"<button onclick=\"location='" + backUrl + "'\">Back</button>");

    public static string ErrorPage(string msg) =>
        "<!DOCTYPE html><html><head><meta charset=\"utf-8\"><style>" + Css + "</style></head>" +
        "<body><div class=\"card\"><h3>Error</h3><p>" + H(msg) + "</p></div><button onclick=\"history.back()\">Back</button></body></html>";

    public static string RedirectPage(string url) =>
        "<!DOCTYPE html><html><head><meta charset=\"utf-8\"><meta http-equiv=\"refresh\" content=\"0;url=" + H(url) + "\">" +
        "</head><body>Redirecting... <a href=\"" + H(url) + "\">click here</a></body></html>";
}
