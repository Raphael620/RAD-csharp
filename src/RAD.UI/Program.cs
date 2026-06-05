using Aprillz.MewUI;

namespace RAD.UI;

class Program
{
    static void Main(string[] args)
    {
        Win32Platform.Register();
        Direct2DBackend.Register();

        var window = new MainWindow().Build();
        Application.Run(window);
    }
}
