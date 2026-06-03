add_cus_dep('syx', 'los', 0, 'makesymbols');
sub makesymbols {
    my ($base) = @_;
    system("makeindex -o \"$base.los\" \"$base.syx\"");
}

add_cus_dep('abx', 'lab', 0, 'makeabbreviations');
sub makeabbreviations {
    my ($base) = @_;
    system("makeindex -o \"$base.lab\" \"$base.abx\"");
}