#! /usr/bin/perl
##########################################################
# pdbnmd.pl
# generate input pdb/psf.
##########################################################
use strict;
use warnings;
#use Fatal qw(open close);
use File::Basename;
use Getopt::Long;

my $home = $ENV{'HOME'};

my $wdir       = "";
my $psfgen     = "psfgen";
my $nmin       = "nmin";
my $ertf       = "$home/pdb_prep/mol-prms/pdbamino.rtf";
my $eprm       = "$home/pdb_prep/mol-prms/parm.prm";
my $nsteps     = 1000;
my $elink      = "";
my $smod       = "";
my $xplor_psf  = 0;
my $auto_disu  = 1;
my $clean      = 1;
my $osuffix    = "_nmin";
my $genosuffix = "_ngen";
my @tofix;
my $nfix       = 0;
my $mod;
my $modid;
my $lastmodid  = "nan";
my $firstpatch = "";

GetOptions(
    "link=s"        => \$elink,
    "smod=s"        => \$smod,
    "rtf=s"         => \$ertf,
    "prm=s"         => \$eprm,
    "wdir=s"        => \$wdir,
    "first=s"       => \$firstpatch,
    "psfgen=s"      => \$psfgen,
    "nmin=s"        => \$nmin,
    "nsteps=i"      => \$nsteps,
    "xplor-psf"     => sub { $xplor_psf = 1 },
    "no-auto-disu"  => sub { $auto_disu = 0 },
    "dont-minimize" => sub { $nsteps = 0; $osuffix = "_ngen" },
    "dont-clean"    => sub { $clean = 0 },
    "osuffix:s"     => \$osuffix,
);

if ($nsteps == 0) {
    $osuffix = "_ngen";
}

if ( scalar @ARGV < 2 ) {
    print_short_args();
    exit 1;
}
my $pdb    = shift @ARGV;
my @chains = @ARGV;

# check name
my ( $pdbname, $pdbdir, $pdbsuffix ) = fileparse( $pdb, qr/\.[^.]*/ );
if ( $pdbsuffix eq ".pdb" ) {
    $pdb = substr( $pdb, 0, length($pdb) - 4 );
}

# process first patch info
my %fpa;

my @fpat = split( /,/, $firstpatch );
my $pa;
my $nfpat = @fpat;
if ($nfpat>1)
{
    $pa = shift @fpat;
    foreach my $fch (@fpat)
    {
       $fpa{$fch} = $pa;
    }
}



# deal with ? chain id
my $ncho = -1;
my @nk;
my $mi;
foreach my $cho (@chains) {
    $ncho = $ncho + 1;
    if ( $cho eq "?" ) {
        @nk = ( @nk, $ncho );
    }
}

if (@nk) {
    for ( $mi = @nk - 1 ; $mi >= 0 ; $mi-- ) {
        splice( @chains, $nk[$mi], 1 );
    }

    my @filss = glob("$wdir$pdb-?.*.pdb");
    chomp @filss;
    @filss = sort { substr( $a, -9 ) cmp substr( $b, -9 ) } @filss;
    foreach my $lin (@filss) {
        my $nu = length($lin) - 10;
        my $no = length( $wdir . $pdb . "-" );
        $lin = substr( $lin, $no, $nu - $no + 1 );
    }
    my %chainh = map { $_, 1 } @chains;    # get unique elements
    for (@filss) {
        next if $chainh{$_}++;
        push @chains, $_;
    }
}

my $nlink = 0;
my @linkl;

if ($elink) {
    @linkl = split( /,/, $elink );
    $nlink = @linkl;
    if ( $nlink - int( $nlink / 5 ) * 5 != 0 ) {
        print "inconsistent definition of link pairs\n";
        print "$elink\n";
        exit 2;
    }
    $nlink = $nlink / 5;
}


my @ltype;
my @sds1;
my @sdsv1;
my @sds2;
my @sdsv2;
my @rds1;
my @rds2;
my $i;
for ( $i = 0 ; $i < $nlink ; $i++ ) {
    @ltype = ( @ltype, shift @linkl );
    @sds1  = ( @sds1,  shift @linkl );
    @sdsv1 = ( @sdsv1, "nan" );
    @rds1  = ( @rds1,  shift @linkl );
    @sds2  = ( @sds2,  shift @linkl );
    @sdsv2 = ( @sdsv2, "nan" );
    @rds2  = ( @rds2,  shift @linkl );
}

if ($auto_disu) {
	my $min_d = 1.28925530695; #mean disulfide distance +- 2 stdev from whole pdb
	my $max_d = 2.82114477374;
	my @cysteine_SGs;
	foreach my $cho (@chains) {
    	$cho = lc($cho);
        my $nfil = 0;
    	foreach my $chain_file (glob("$wdir$pdb-$cho.*.pdb")) {
			my @lines = ` grep 'SG  CYS' '$chain_file'`;
			foreach my $line (@lines) {
				push @cysteine_SGs, {chain => uc($cho), line => $line, seg => $smod.$cho.$nfil};
			}
			$nfil = $nfil +1;
		}
	}
	my $num_SG = scalar @cysteine_SGs;
	for (my $i = 0; $i < $num_SG; $i++) {
		for (my $j = $i+1; $j < $num_SG; $j++) {
			my $x1 = substr($cysteine_SGs[$i]{line}, 30, 8);
			my $y1 = substr($cysteine_SGs[$i]{line}, 38, 8);
			my $z1 = substr($cysteine_SGs[$i]{line}, 46, 8);
			my $x2 = substr($cysteine_SGs[$j]{line}, 30, 8);
			my $y2 = substr($cysteine_SGs[$j]{line}, 38, 8);
			my $z2 = substr($cysteine_SGs[$j]{line}, 46, 8);

			my $dist = sqrt( ($x1-$x2)**2 + ($y1-$y2)**2 + ($z1-$z2)**2 );
			if ( ($dist < $max_d) && ($dist > $min_d) ) {
				$nlink++;
				my $res1 = substr($cysteine_SGs[$i]{line}, 22, 5);
				my $res2 = substr($cysteine_SGs[$j]{line}, 22, 5);
				$res1 =~ s/\s+//g;
				$res2 =~ s/\s+//g;
				push @linkl, ('DISU', $cysteine_SGs[$i]{chain}, $res1,  $cysteine_SGs[$j]{chain}, $res2);
				@ltype = ( @ltype, 'DISU' );
				@sds1  = ( @sds1,  $cysteine_SGs[$i]{chain} );
				@sdsv1 = ( @sdsv1, $cysteine_SGs[$i]{seg} );
				@rds1  = ( @rds1,  $res1 );
				@sds2  = ( @sds2,  $cysteine_SGs[$j]{chain} );
				@sdsv2 = ( @sdsv2, $cysteine_SGs[$j]{seg} );
				@rds2  = ( @rds2,  $res2 );
			}
		}
	}
}

my $rtf     = $ertf;
my $oscript = "$wdir$$.inp";

my $cha;

open OSCRIPT, '>', $oscript;
print OSCRIPT "# read protein\n";
print OSCRIPT "# multiple chains with breaks\n";
print OSCRIPT "#\n";

print OSCRIPT "# RTF FILE\n";
print OSCRIPT "topology $rtf\n";

print OSCRIPT "# GENERATE CHAINS\n";
foreach my $cho (@chains) {
    $cho = lc($cho);
    my @files = glob("$wdir$pdb-$cho.*.pdb");
    chomp @files;
    my $nfil = 0;
    if ( (length($smod) + length($cho) ) > 3) { #illegal, cause psfgen crash
        my $diff = (length($smod) + length($cho) ) - 3;
        my $newcho = substr($cho, $diff);
        $cha = $smod . $newcho;
    } else {
        $cha = $smod . $cho;
    }
    foreach my $fil (@files) {
        $pa="NONE";
        if (exists($fpa{$cha}))
        {
           $pa=$fpa{$cha};
        }

        print OSCRIPT "segment $cha$nfil {first $pa; last NONE; pdb $fil }\n";
        $nfil = $nfil + 1;
    }
}

print OSCRIPT "#PATCH NONCONTINUOUS CHAINS\n";
foreach my $cho (@chains) {
    $cho = lc($cho);
    my @files = glob("$wdir$pdb-$cho.*.pdb");
    chomp @files;
    $cha = $smod . $cho;
    my $nfil = 0;
    my $sc   = "nan";
    my $cx;
    my $cy;
    my $cz;
    my $rc = "nan";
    my $rcn;
    my $sn;
    my $nx;
    my $ny;
    my $nz;
    my $rn;
    my $rnn;
    my $goodc = 0;
    my $goodn = 0;

    foreach my $fil (@files) {
        open my $PDB, '<', $fil;
        my @pdblines = <$PDB>;
        chomp @pdblines;
        close $PDB;
        my $line;
        my $fres  = -999;
        my $lres  = -999;
        my $fresi = "-999 ";
        my $lresi = "-999 ";
        foreach my $inline (@pdblines) {
            my $record = substr( $inline, 0, 6 );
            if ( $record eq "ATOM  " || $record eq "HETATM" ) {
                $lres  = substr( $inline, 22, 4 );
                $lresi = substr( $inline, 22, 5 );
                $mod   = substr( $inline, 26, 1 );
                if ( $mod ne " " ) {
                    $modid = uc($cha) . $nfil . " " . $lres . " " . $mod;
                    if ( $modid ne $lastmodid ) {
                        $tofix[$nfix] = $modid;
                        $nfix++;
                        $lastmodid = $modid;
                    }
                }
                if ( $fres == -999 ) {
                    $fres  = $lres;
                    $fresi = $lresi;
                }
            }
        }
        $goodn = 0;
        if ( $goodc == 1 ) {
            foreach my $inline (@pdblines) {
                my $record = substr( $inline, 0, 6 );
                if ( $record eq "ATOM  " || $record eq "HETATM" ) {
                    my $nnam = substr( $inline, 12, 4 );
                    if ( $nnam eq " N  " ) {
                        $line  = $inline;
                        $goodn = 1;
                        last;
                    }
                }
            }
            if ( $goodn == 1 ) {
                $nx = substr( $line, 30, 8 );
                $ny = substr( $line, 38, 8 );
                $nz = substr( $line, 46, 8 );

                $nx = $nx - $cx;
                $ny = $ny - $cy;
                $nz = $nz - $cz;
                my $dr = sqrt( $nx * $nx + $ny * $ny + $nz * $nz );
                if ( $dr <= 1.4 ) {
                    $rn  = substr( $line, 17, 3 );
                    $rnn = substr( $line, 22, 4 );
                    $rnn =~ /(\w+)/;
                    $rnn = $1;

                    $sn = $cha . $nfil;
                    if ( $rc eq "GLY" ) {
                        if ( $rn eq "GLY" ) {
                            print OSCRIPT "patch JOGG $sc:$rcn $sn:$rnn\n";
                        }
                        elsif ( $rn eq "PRO" ) {
                            print OSCRIPT "patch JOGP $sc:$rcn $sn:$rnn\n";
                        }
                        else {
                            print OSCRIPT "patch JOGA $sc:$rcn $sn:$rnn\n";
                        }
                    }
                    elsif ( $rn eq "PRO" ) {
                        print OSCRIPT "patch JOAP $sc:$rcn $sn:$rnn\n";
                    }
                    elsif ( $rn eq "GLY" ) {
                        print OSCRIPT "patch JOAG $sc:$rcn $sn:$rnn\n";
                    }
                    else {
                        print OSCRIPT "patch JOAA $sc:$rcn $sn:$rnn\n";
                    }
                }
            }
        }
        $goodc    = 0;
        @pdblines = reverse(@pdblines);
        foreach my $inline (@pdblines) {
            my $record = substr( $inline, 0, 6 );
            if ( $record eq "ATOM  " || $record eq "HETATM" ) {
                my $cnam = substr( $inline, 12, 4 );
                if ( $cnam eq " C  " ) {
                    $line  = $inline;
                    $goodc = 1;
                    last;
                }
            }
        }

        if ( $goodc == 1 ) {
            $sc = $cha . $nfil;
            $rcn = substr( $line, 22, 4 );
            $rcn =~ /(\w+)/;
            $rcn = $1;
            $rc  = substr( $line, 17, 3 );
            $cx  = substr( $line, 30, 8 );
            $cy  = substr( $line, 38, 8 );
            $cz  = substr( $line, 46, 8 );
        }

        for ( $i = 0 ; $i < $nlink ; $i++ ) {
            foreach my $inline (@pdblines) {
                my $record = substr( $inline, 0, 6 );
                if ( $record eq "ATOM  " || $record eq "HETATM" ) {
                    my $pres = substr( $inline, 22, 5 );
                    $pres =~ /(\w+)/;
                    $pres = $1;
                    my $acho = uc($cho);

                    if ( $sds1[$i] eq $acho and $rds1[$i] eq $pres and $sdsv1[$i] eq "nan" ) {
                        $sdsv1[$i] = "$cha$nfil";
                    }
                    if ( $sds2[$i] eq $acho and $rds2[$i] eq $pres and $sdsv2[$i] eq "nan" ) {
                        $sdsv2[$i] = "$cha$nfil";
                    }
                }
            }
        }

        $nfil = $nfil + 1;
    }
}

print OSCRIPT "#PATCH LINKAGE BONDS\n";
for ( $i = 0 ; $i < $nlink ; $i++ ) {
    if ( $sdsv1[$i] eq "nan" or $sdsv2[$i] eq "nan" ) {
        print "ERROR in link patching: cannot match a segment\n";
        print "$sds1[$i]:$rds1[$i]:$sdsv1[$i] $sds2[$i]:$rds2[$i]:$sdsv2[$i]\n";
        exit 2;
    }
    my $so1 = $rds1[$i];
    $so1 =~ /(\d+)/;
    $so1 = $1;
    $rds1[$i] = $so1;
    my $so2 = $rds2[$i];
    $so2 =~ /(\d+)/;
    $so2 = $1;
    $rds2[$i] = $so2;
    print OSCRIPT "patch $ltype[$i] $sdsv1[$i]:$rds1[$i] $sdsv2[$i]:$rds2[$i]\n";
}

print OSCRIPT "#READ COORDINATES\n";
foreach my $cho (@chains) {
    $cho = lc($cho);
    my @files = glob("$wdir$pdb-$cho.*.pdb");
    chomp @files;
    my $nfil = 0;
    if ( (length($smod) + length($cho) ) > 3) { #illegal, cause psfgen crash
        my $diff = (length($smod) + length($cho) ) - 3;
        my $newcho = substr($cho, $diff);
        $cha = $smod . $newcho;
    } else {
        $cha = $smod . $cho;
    }
    foreach my $fil (@files) {
        print OSCRIPT "coordpdb $fil $cha$nfil\n";
        $nfil = $nfil + 1;
    }
}
print OSCRIPT "guesscoord\n";

print OSCRIPT "writepsf charmm $wdir$pdb$osuffix.psf\n";
if ( $xplor_psf == 1 ) {
    print OSCRIPT "writepsf x-plor $wdir$pdb$osuffix\_xplor.psf\n";
}
print OSCRIPT "writepdb $wdir$pdb$genosuffix.pdb\n";

close OSCRIPT;
my $out = $oscript . "\.out";
system "$psfgen < $oscript>$out";

#restore inserted residues (eg 32A, 32B, etc.)
open my $PSF, '<', "$wdir$pdb$osuffix.psf";
my @psflines = <$PSF>;
chomp @psflines;
close $PSF;

my $nl = @psflines;
my $i1;

my $istart = 0;
for ( $i1 = 0 ; $i1 < $nl ; $i1++ ) {
    if ( $psflines[$i1] =~ /!NATOM/ ) { last; }
}
if ( $psflines[$i1] =~ /!NATOM/ ) {
    my $psfl = $psflines[$i1];
    my @spli = split( / /, $psfl );
    while ( $spli[0] !~ m/[0-9]/ ) { shift @spli; }
    my $natoms = $spli[0] + 0;
    $istart = $i1 + 1;
    for ( $i1 = $istart ; $i1 < $istart + $natoms ; $i1++ ) {
        $psfl = $psflines[$i1];
        my @check = split( / +/, $psfl );
        foreach my $some (@tofix) {
            my @check1 = split( / +/, $some );
            if ( $check1[0] eq $check[2] and $check1[1] eq $check[3] ) {
                substr( $psfl, 14 + length($check1[1]), 1 ) = $check1[2];
                $psflines[$i1] = $psfl;
            }
        }
    }
}

open $PSF, '>', "$wdir$pdb$osuffix.psf";
foreach my $psfl (@psflines) {
    print $PSF "$psfl\n";
}
close $PSF;

if ( $xplor_psf == 1 ) {
    open my $PSF, '<', "$wdir$pdb$osuffix\_xplor.psf";
    my @psflines = <$PSF>;
    chomp @psflines;
    close $PSF;

    my $nl = @psflines;
    my $i1;

    my $istart = 0;
    for ( $i1 = 0 ; $i1 < $nl ; $i1++ ) {
        if ( $psflines[$i1] =~ /!NATOM/ ) { last; }
    }
    if ( $psflines[$i1] =~ /!NATOM/ ) {
        my $psfl = $psflines[$i1];
        my @spli = split( / /, $psfl );
        while ( $spli[0] !~ m/[0-9]/ ) { shift @spli; }
        my $natoms = $spli[0] + 0;
        $istart = $i1 + 1;
        for ( $i1 = $istart ; $i1 < $istart + $natoms ; $i1++ ) {
            $psfl = $psflines[$i1];
            my @check = split( / +/, $psfl );
            foreach my $some (@tofix) {
                my @check1 = split( / +/, $some );
                if ( $check1[0] eq $check[2] and $check1[1] eq $check[3] ) {
                    substr( $psfl, 14 + length($check1[1]), 1 ) = $check1[2];
                    $psflines[$i1] = $psfl;
                }
            }
        }
    }
    open $PSF, '>', "$wdir$pdb$osuffix\_xplor.psf";
    foreach my $psfl (@psflines) {
        print $PSF "$psfl\n";
    }
    close $PSF;
}

open my $PDB, '<', "$wdir$pdb$genosuffix.pdb";
my @pdblines = <$PDB>;
chomp @pdblines;
close $PDB;
$nl = @pdblines;
for ( $i1 = 0 ; $i1 < $nl ; $i1++ ) {
    my $inline = $pdblines[$i1];
    my $record = substr( $inline, 0, 6 );
    if ( $record eq "ATOM  " || $record eq "HETATM" ) {
        my @check = split( / +/, $inline );
        foreach my $some (@tofix) {
            my @check1 = split( / +/, $some );
            if ( $check1[0] eq $check[11] and $check1[1] eq $check[5] ) {
                substr( $inline, 26, 1 ) = $check1[2];
                $pdblines[$i1] = $inline;
            }
        }
    }
}

open $PDB, '>', "$wdir$pdb$genosuffix.pdb";
foreach my $inline (@pdblines) {
    print $PDB "$inline\n";
}
close $PDB;

my $fixed_pdb = "$wdir$pdb$genosuffix.fixed.pdb";
if ( $nsteps > 0 ) {

    #generate fixed residue file on assumption of 0 in occupancy column
    open my $GENPDB, '<', "$wdir$pdb$genosuffix.pdb";
    open my $FIXPDB, '>', $fixed_pdb;
    while (<$GENPDB>) {
        print $FIXPDB $_ unless ( /^ATOM/ && substr( $_, 55, 5 ) eq ' 0.00' );
    }
    close $GENPDB;
    close $FIXPDB;
    system
"$nmin $wdir$pdb$osuffix.psf $eprm $ertf $wdir$pdb$genosuffix.pdb $fixed_pdb $nsteps > $wdir$pdb$osuffix.pdb";
}

if ($clean) {
    unlink($oscript);
    unlink($out);
    unlink("$wdir$pdb\_$$.crd");
    unlink("$wdir$pdb\_$$.ener");
    unlink("$wdir$pdb\_$$.ic");
    unlink($fixed_pdb);
    if ( $osuffix ne $genosuffix ) {
        unlink("$wdir$pdb$genosuffix.pdb");
    }
}

sub print_short_args {
    printf( "usage: %s options PDB_code ch1 ch2 ...\n", $0 );
    printf("options: \n");
    printf("         --link seg1,res1,seg2,res2,...\n");
    printf("         --first 5ter,chain1,chain2,...\n");
    printf("         --smod segment_prefix\n");
    printf("         --wdir wdir\n");
    printf("         --psfgen psfgen\n");
    printf("         --nmin nmin\n");
    printf("         --prm prm\n");
    printf("         --rtf rtf\n");
    printf("         --nsteps nsteps\n");
    printf("         --xplor-psf\n");
    printf("         --dont-clean\n");
    printf("         --dont-minimize\n");
    printf("         --osuffix=s (default: _nmin)\n");
}
