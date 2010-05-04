        subroutine sdgemm(M, A, B, C)
c
c       .. Parameters ..
        integer M
        double precision A(M,M)
        double precision B(M,M)
        double precision C(M,M)
        double precision Ctemp
c
c       .. Local variables ..
        integer i
        integer j
        integer k
c
c       .. Nested loops ..
        do i=1,M
          do j=1,M
            Ctemp = C(i,j)
            do k=1,M
              Ctemp = Ctemp + A(i,k)*B(k,j)
            end do
            C(i,j) = Ctemp
          end do
        end do
c
        return
c
        end
