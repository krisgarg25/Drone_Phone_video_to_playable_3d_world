$ErrorActionPreference = 'Stop'
$pptx = 'c:\Users\krisg\Desktop\Drone to 3d mesh\ppt\SIH26158_ONEPASS_Idea_Deck.pptx'
$outDir = 'c:\Users\krisg\Desktop\Drone to 3d mesh\ppt\_render'
if (Test-Path $outDir) { Remove-Item $outDir -Recurse -Force }
New-Item -ItemType Directory -Path $outDir | Out-Null

$app = New-Object -ComObject PowerPoint.Application
try {
    $pres = $app.Presentations.Open($pptx, $true, $false, $false)
    Write-Output ("slides: " + $pres.Slides.Count)
    foreach ($s in $pres.Slides) {
        $f = Join-Path $outDir ("slide{0:d2}.png" -f $s.SlideIndex)
        # 2 = ppSlideSaveAsPNG, width 1600
        $s.Export($f, "PNG", 1600, 900)
        Write-Output ("exported " + $f)
    }
    $pres.Close()
} finally {
    $app.Quit()
    [System.Runtime.Interopservices.Marshal]::ReleaseComObject($app) | Out-Null
}
Get-ChildItem $outDir | Select-Object Name, Length | Format-Table -AutoSize | Out-String
